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

    Eigen::Matrix4d e_Twc;
    Eigen::Matrix4d base_e_Twc;
    int cnter_fused = 0;

    float T_wb[4 * 4] = {0};  //基坐标
    float T_bw[4 * 4] = {0}; //Tcw

    int fuse_cnt = 0;
    string fixed_time;

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
            ReadDepth(depth_mat, depth_gt_mat, depth, curr_scale);

            float T_wc[4 * 4] = {0};
            cnter_fused++;

            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    T_wc[r * 4 + c] = e_Twc(r, c);

            float T_b_curr[16] = {0};
            multiply_matrix(T_bw, T_wc, T_b_curr); // Tc1w * Twc2 = Tc1c2

            fuse_cnt++;
        }
        else
            continue;
        if (data_manager.shouldQuit())
            break;
    }
}

