#include "parameters.h"

param Cfgparam;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr curr_scene;


void readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    cout << "start loading undistort parameters......" << endl;
    Cfgparam.apply_scale = fsSettings["apply_scale"];


    string gt_path;
    fsSettings["gt_path"] >> gt_path;

    fsSettings["est_depth_path"] >> Cfgparam.est_depth_path;
    Cfgparam.est_depth_path = Cfgparam.est_depth_path + "/depth";

    fsSettings["additional_name"] >> Cfgparam.additional_name;
    fsSettings["video_path"] >> Cfgparam.color_path;


    if (Cfgparam.apply_scale)  //gazebo sim 数据专用
    {
        Cfgparam.color_path = gt_path + "/imgs";
        Cfgparam.depth_gt_path = gt_path + "/depth";
        Cfgparam.gt_pose_file = gt_path + "/camera_pose.txt";
        Cfgparam.depth_gt_mfactor = fsSettings["depth_gt_mfactor"];
        double depth_decrease600 = fsSettings["depth_decrease600"];
        double depth_decrease800 = fsSettings["depth_decrease800"];
        double depth_decrease1000 = fsSettings["depth_decrease1000"];
        double depth_decrease1200 = fsSettings["depth_decrease1200"];
        double depth_decrease1400 = fsSettings["depth_decrease1400"];
        double depth_decrease1600 = fsSettings["depth_decrease1600"];

        string mask = gt_path.substr(gt_path.find_last_of("/") + 1,
                                     5).c_str();

        if (mask == "curve")
        {
            Cfgparam.depth_gt_dfactor = 700;
            ifstream data_annotation(gt_path + "/data_annotation.txt");
            std::string data_line_pose;
            std::string ano_type;
            while (true)
            {
                if (data_annotation.eof())
                    exit(-1);
                std::getline(data_annotation, data_line_pose);
                std::istringstream annotation(data_line_pose);
                annotation >> ano_type >> Cfgparam.depth_gt_dfactor;
                if (ano_type == "dfactor:")
                    break;
            }
        }
        else
        {
            double data_height = atof(Cfgparam.depth_gt_path.substr(Cfgparam.depth_gt_path.find_last_of("H") + 1,
                                                                    Cfgparam.depth_gt_path.find_last_of("L")
                                                                        - Cfgparam.depth_gt_path.find_last_of("H") - 2)
                                          .c_str());

            switch (int(data_height))
            {
                case 600:Cfgparam.depth_gt_dfactor = depth_decrease600;
                    break;
                case 800:Cfgparam.depth_gt_dfactor = depth_decrease800;
                    break;
                case 1000:Cfgparam.depth_gt_dfactor = depth_decrease1000;
                    break;
                case 1200:Cfgparam.depth_gt_dfactor = depth_decrease1200;
                    break;
                case 1400:Cfgparam.depth_gt_dfactor = depth_decrease1400;
                    break;
                case 1600:Cfgparam.depth_gt_dfactor = depth_decrease1600;
                    break;
            }
        }
    }

    fsSettings["est_pose_file"] >> Cfgparam.est_pose_file;
    fsSettings["pc_save_path"] >> Cfgparam.pc_save_path;

    Cfgparam.depth_result_factor = fsSettings["depth_result_factor"];
    Cfgparam.tsdf_threshold = fsSettings["tsdf_threshold"];
    Cfgparam.factor_voxel = fsSettings["factor_voxel"];

    Cfgparam.fx = fsSettings["Camera.fx"];
    Cfgparam.fy = fsSettings["Camera.fy"];
    Cfgparam.cx = fsSettings["Camera.cx"];
    Cfgparam.cy = fsSettings["Camera.cy"];

    Cfgparam.voxelField_rate_x = fsSettings["voxelField_rate_x"];
    Cfgparam.voxelField_rate_y = fsSettings["voxelField_rate_y"];
    Cfgparam.voxelField_rate_z = fsSettings["voxelField_rate_z"];

    Cfgparam.k1 = fsSettings["Camera.k1"];
    Cfgparam.k2 = fsSettings["Camera.k2"];
    Cfgparam.k3 = fsSettings["Camera.p1"];

    Cfgparam.r1 = fsSettings["Camera.p2"];
    Cfgparam.r2 = 0;
    Cfgparam.r2 = fsSettings["Camera.p3"];

    Cfgparam.TruncationScale = fsSettings["TruncationScale"];

    Cfgparam.GridSize_x = fsSettings["GridSize_x"];
    Cfgparam.GridSize_y = fsSettings["GridSize_y"];
    Cfgparam.GridSize_z = fsSettings["GridSize_z"];

    Cfgparam.n_line2giveup = fsSettings["n_line2giveup"];
    Cfgparam.incremental_show = fsSettings["incremental_show"];

    double width = fsSettings["Camera.width"];
    double height = fsSettings["Camera.height"];

    Cfgparam.img_size = cv::Size(width, height);
    Cfgparam.save_pcd = fsSettings["save_pcd"];
    Cfgparam.start_ID = fsSettings["start_time"];

    cout << "depth_gt_dfactor = " << Cfgparam.depth_gt_dfactor << endl;
    cout << "depth_gt_mfactor = " << Cfgparam.depth_gt_mfactor << endl;

    fsSettings.release();
    Cfgparam.showParams();
}
