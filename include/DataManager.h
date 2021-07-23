//
// Created by lab on 2020-09-07.
//

#ifndef OPTFLOW_TRACK_V1_DATAMANAGER_H
#define OPTFLOW_TRACK_V1_DATAMANAGER_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "parameters.h"
#include "getfile.h"


using namespace cv;
using namespace std;
using namespace Eigen;

class DataManager
{
public:
    DataManager(param &Cfgparam);
    void
    getAllData();
    void
    readPoses(string file_path, vector<double> &Pose_Timestamp,std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &est_Poses_Twc,
              bool set_first_pose_identity
    );
    void
    get_and_align_poses();
    bool
    grebFrame(cv::Mat &color_raw, cv::Mat &depth, cv::Mat &depth_gt, Matrix4d &Twc,double & curr_scale,double &curr_time_stamp);
    void
    getDirImages(string data_path, vector<string> &vstrImages_tmp);
    bool
    shouldQuit()
    {
            return (
                color_idx >= nImagesTol ||
                    pose_idx >= est_Poses_Twc.size() ||
                    depth_idx >= Depth_Maps.size()
            );

    };

//	获取图像，深度图
    string color_path;
    string depth_path;
    string depth_gt_path;

    VideoCapture capture;
    double fps = 30.0;
    double nImagesTol;
    int color_idx = 0;
    int pose_idx = 0;
    int depth_idx = 0;
    int start_ID = 0;

    double depth_gt_mfactor;
    double depth_gt_dfactor;
    vector<string> Color_Imgs;
    vector<string> Depth_Maps;
    vector<string> Depth_gt_Maps;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> est_Poses_Twc;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_Poses_Twc;

    vector<double> est_Pose_Timestamp;
    vector<double> gt_Pose_Timestamp;
    vector<double> Depth_Timestamps;
    vector<double> Depth_gt_Timestamps;
    vector<double> scale_gt_div_est;

};


#endif //OPTFLOW_TRACK_V1_DATAMANAGER_H
