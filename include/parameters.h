#pragma once
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>

//多线程库
#include <omp.h>

#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "tic_toc.h"



using namespace std;
using namespace cv;





enum viz_status{show_kf,show_kf_inc,show_targ, show_targ_inc};



struct param {
//
    int apply_scale;

    int incremental_show;
    int n_line2giveup;
    string color_path;
    string depth_gt_path;
    string est_pose_file;
    string gt_pose_file;
    std::string est_depth_path;
    string pc_save_path;

    double depth_result_factor;
    double tsdf_threshold;
    std::string data_formate;
    std::string additional_name;

    double TruncationScale;

    float voxelField_rate_x;
    float voxelField_rate_y;
    float voxelField_rate_z;

    double GridSize_x;
    double GridSize_y;
    double GridSize_z;

    int start_ID = 0;

    double depth_gt_mfactor;
    double depth_gt_dfactor;
    int save_pcd = 0;

    cv::Size img_size;
    double factor_voxel;
    double fx;
    double fy;
    double cx;
    double cy;

    double k1;
    double k2;
    double k3;
    double r1;
    double r2;


    void showParams(){
        cout << "color_path:" << color_path << endl;
        cout << "est_depth_path:" << est_depth_path << endl;
        cout<<"width:"<<img_size.width<<endl;
        cout<<"height:"<<img_size.height<<endl;
        cout<<"fx:"<<fx<<endl;
        cout<<"fy:"<<fy<<endl;
        cout<<"cx:"<<cx<<endl;
        cout<<"cy:"<<cy<<endl;
        cout<<"k1:"<<k1<<endl;
        cout<<"k2:"<<k2<<endl;
        cout<<"k3:"<<k3<<endl;
        cout<<"r1:"<<r1<<endl;
        cout<<"r2:"<<r2<<endl;
    }

};


extern param Cfgparam;
extern pcl::PointCloud<pcl::PointXYZRGB>::Ptr curr_scene;

void readParameters(std::string config_file);
