//
// Created by will on 19-10-17.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "Timer.h"
#include "tic_toc.h"
#include "CUDATSDFIntegrator.h"
#include "Reader.h"
#include "parameters.h"
#include "DataManager.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace std;


int fused_scene = 0;  //一共融合了的场景

enum STATE { scene_reflash, new_scene, reflashed };

STATE curr_state = reflashed;

mutex mtx;

int
fuse_depth_map(int first_frame_idx, Eigen::Matrix4d &Twc);

int
display_thd()
{

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->addPointCloud(curr_scene, "curr_scene");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "curr_scene"); // 设置点云大小
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);

        if (curr_state == scene_reflash)
        {
            mtx.lock();
//            viewer->removeAllPointClouds();  // 移除当前所有点云
//            viewer->addPointCloud(curr_scene, "curr_scene");
            viewer->updatePointCloud(curr_scene, "curr_scene");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     6,
                                                     "curr_scene"); // 设置点云大小
            mtx.unlock();
            curr_state = reflashed;
            cout << "scene reflashed!" << endl;
        }

        if (curr_state == new_scene)
        {
            mtx.lock();
            viewer->removeCoordinateSystem("curr_scene");  // 移除当前所有点云
            viewer->addPointCloud(curr_scene, "history_scene" + to_string(fused_scene));

            mtx.unlock();
            curr_state = reflashed;
            cout << "new reflashed! --------------- " << endl;
        }
        //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

int
main(int argc, char *argv[])
{
//  设置参数
    if (argc == 1)
        argv[1] = "../config/fusion_config_my.yaml";

    readParameters(argv[1]);

//  准备数据
    DataManager data_manager(Cfgparam);
    data_manager.getAllData();

    curr_scene.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    curr_scene->is_dense = false;

    std::thread thd_Draw;
    if (Cfgparam.incremental_show)
        thd_Draw = std::thread(display_thd);

    Eigen::Matrix4d e_Twc;
    Eigen::Matrix4d base_e_Twc;
    int cnter_fused = 0;
    bool finished = false;

//        开启一次新的融合
    CUDATSDFIntegrator Fusion;

    float T_wb[4 * 4] = {0};  //基坐标
    float T_bw[4 * 4] = {0}; //Tcw

    int fuse_cnt = 0;
    string fixed_time;
//    uchar3 *color = new uchar3[Cfgparam.img_size.width * Cfgparam.img_size.height];
//    memset(color, 255, sizeof(uchar3) * Cfgparam.img_size.width * Cfgparam.img_size.height); //清零

    float depth[Cfgparam.img_size.width * Cfgparam.img_size.height];
    TicToc timer;
    while (!data_manager.shouldQuit())
    {
        double curr_time_stamp = 0;
        cv::Mat depth_mat;
        cv::Mat depth_gt_mat;
        cv::Mat color_img;
        double curr_scale;
        timer.tic();
        bool has_results =
            data_manager.grebFrame(color_img, depth_mat, depth_gt_mat, e_Twc, curr_scale, curr_time_stamp);

        if (has_results && !color_img.empty() && !depth_mat.empty())
        {
//            设置参考帧(基坐标系)
            if (!Fusion.is_initialized || Fusion.need_reset)
            {
                if (sum(depth_mat)[0] < 10000.0)
                    continue;

                fixed_time = to_string(curr_time_stamp);

                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        T_wb[r * 4 + c] = e_Twc(r, c);

//              设置当前帧为融合的参考帧(基参考系),保存当前参考帧到世界的位姿
                invert_matrix(T_wb, T_bw);
                base_e_Twc = e_Twc;
                Fusion.Initialize(depth_mat, depth_gt_mat, e_Twc, depth, curr_scale);
            }
            else
            {
                ReadDepth(depth_mat, depth_gt_mat, depth, curr_scale);
            }

            float T_wc[4 * 4] = {0};
            cnter_fused++;

//            cv::cvtColor(color_img, color_img, CV_BGR2RGB);
//            memcpy(color, color_img.data, sizeof(uchar3) * Cfgparam.img_size.width * Cfgparam.img_size.height);

            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    T_wc[r * 4 + c] = e_Twc(r, c);

            float T_b_curr[16] = {0};
            multiply_matrix(T_bw, T_wc, T_b_curr); // Tc1w * Twc2 = Tc1c2

            Fusion.integrate(depth, color_img.data, T_b_curr);  //输入深度数组float类型的，彩色图像，当前帧到参考基矩阵的转换矩阵

            std::cout << "Frame Index:" << Fusion.FrameId << " of " << data_manager.nImagesTol
                      << " time: " << curr_time_stamp
                      << endl;

            fuse_cnt++;
        }
        else
            continue;

        if (data_manager.shouldQuit())
        {
            finished = true;
            break;
        }
        else if (Cfgparam.incremental_show)
            curr_state = scene_reflash;

        if (Fusion.need_reset)
        {
            Fusion.SaveVoxelGrid2SurfacePointCloud_1(Cfgparam.tsdf_threshold, 0.0, T_wb);
            if(!curr_scene->empty())
                pcl::io::savePCDFileBinary(Cfgparam.pc_save_path + "/fused_est" + fixed_time + ".pcd", *curr_scene);
        }

    }

    Fusion.SaveVoxelGrid2SurfacePointCloud_1(Cfgparam.tsdf_threshold, 0.0, T_wb);

    if(!curr_scene->empty())
        pcl::io::savePCDFileBinary(Cfgparam.pc_save_path + "/fused_est" + fixed_time + ".pcd", *curr_scene);

    std::cout << "new scene point num: " << curr_scene->size() << " , save PCD cost: " << timer.toc() << " ms."
              << std::endl;
    curr_state = new_scene;
    fused_scene++;
    if (Cfgparam.incremental_show)
    {
        thd_Draw.join();
    }
}

//int
//fuse_depth_map(cv::Mat &color_img, cv::Mat &depth_mat, Eigen::Matrix4d &Twc)
//{
//    string ss;
//    CUDATSDFIntegrator Fusion(ss);
//    int base_frame_idx = first_frame_idx;
//    std::ostringstream base_frame_prefix;
//    base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
//    std::string base2world_file = inputPath + "frame-" + base_frame_prefix.str() + ".pose.txt";
//
//    float Twb[4 * 4] = {0};
//    LoadMatrix(base2world_file, Twb, Twc);
//    // Invert base frame camera pose to get world-to-base frame transform
//    float base2world_inv[16] = {0}; //Tcw
//    invert_matrix(Twb, base2world_inv);
//
//    float depth[im_width * im_height];
//    float pose[4 * 4] = {0};
//    uchar3 *color = new uchar3[im_width * im_height];
//
//    for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int) num_frames_fuse; ++frame_idx)
//    {
//        std::ostringstream curr_frame_prefix;
//        curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
//
//        // Path
//        std::string depth_im_file = inputPath + "frame-" + curr_frame_prefix.str() + ".depth.png";
//        std::string color_im_file = inputPath + "frame-" + curr_frame_prefix.str() + ".color.png";
//        std::string pose_file = inputPath + "frame-" + curr_frame_prefix.str() + ".pose.txt";
//        //std::cout << depth_im_file << std::endl;
//        // Read current frame
//        cv::Mat depth_im = cv::imread(depth_im_file, CV_LOAD_IMAGE_UNCHANGED);
//        cv::Mat color_im = cv::imread(color_im_file);
//
//        if (depth_im_file.empty() || color_im_file.empty())
//            continue;
//
//        //std::cout << "Channels: " << color_im.channels() << std::endl;
//        cv::cvtColor(color_im, color_im, CV_BGR2RGBA);
//        memcpy(color, color_im.data, sizeof(uchar3) * im_height * im_width);
//
//        ReadDepth(depth_im, im_width, im_height, depth, depthfactor);
//        Eigen::Matrix4d T;
//        LoadMatrix(pose_file, pose, T);
//
//        float came2base[16] = {0};
//        multiply_matrix(base2world_inv, pose, came2base); // Tc1w * Twc2 = Tc1c2
//        Fusion.integrate(depth, color, came2base);
//    }
//
//    delete[] color;
//    Fusion.SaveVoxelGrid2SurfacePointCloud(0.2, 0.0, Twc, outPath + "/" + to_string(first_frame_idx) + ".pcd");
//    return 0;
//}
#pragma clang diagnostic pop
