//
// Created by lab on 2020-09-07.
//

#include "DataManager.h"

DataManager::DataManager(param &Cfgparam)
{
    depth_gt_mfactor = Cfgparam.depth_gt_mfactor;
    depth_gt_dfactor = Cfgparam.depth_gt_dfactor;
    color_path = Cfgparam.color_path;
    depth_path = Cfgparam.est_depth_path;
    depth_gt_path = Cfgparam.depth_gt_path;
}

void
DataManager::getDirImages(string img_path, vector<string> &vstrImages_tmp)
{
    vstrImages_tmp.clear();

//    读取所有图像
    if (getdir(img_path, vstrImages_tmp) >= 0)
    {
        printf("found %d image files in folder %s!\n",
               (int) vstrImages_tmp.size(),
               img_path.c_str());
    }
    else if (getFile(img_path.c_str(), vstrImages_tmp) >= 0)
    {
        printf("found %d image files in file %s!\n",
               (int) vstrImages_tmp.size(),
               img_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    //    对图像按照时间戳先后排序
    sort(vstrImages_tmp.begin(), vstrImages_tmp.end(), [](string x, string y)
         {
             string s_time = x
                 .substr(x.find_last_of("/") + 1, x.find_last_of(".") - x.find_last_of("/") - 1);
             double a_stamp = atof(s_time.c_str());

             s_time = y
                 .substr(y.find_last_of("/") + 1, y.find_last_of(".") - y.find_last_of("/") - 1);
             double b_stamp = atof(s_time.c_str());

             return a_stamp < b_stamp;
         }
    );
}

void
DataManager::getAllData()
{
    if (Cfgparam.data_formate == "IMAGES")
    {
        getDirImages(color_path, Color_Imgs);
    }
    else
    {
        capture.open(color_path);
        if (!capture.isOpened())
        {
            printf("can not open ...\n");
            return;
        }
        else
        {
            printf("video opened\n");
        }
    }


    getDirImages(depth_path, Depth_Maps);
    nImagesTol = Depth_Maps.size();

    for (int i = 0; i < Depth_Maps.size(); ++i)
    {
        string &depth_path = Depth_Maps[i];
        string s_time = depth_path
            .substr(depth_path.find_last_of("/") + 1,
                    depth_path.find_last_of(".") - depth_path.find_last_of("/") - 1);
        double b_stamp = atof(s_time.c_str());
        Depth_Timestamps.push_back(b_stamp);
    }

    if (Cfgparam.apply_scale)
    {
        getDirImages(depth_gt_path, Depth_gt_Maps);
        for (int i = 0; i < Depth_gt_Maps.size(); ++i)
        {
            string &depth_path = Depth_gt_Maps[i];
            string s_time = depth_path
                .substr(depth_path.find_last_of("/") + 1,
                        depth_path.find_last_of(".") - depth_path.find_last_of("/") - 1);
            double b_stamp = atof(s_time.c_str());
            Depth_gt_Timestamps.push_back(b_stamp);
        }
    }

    get_and_align_poses();

    start_ID = Depth_Timestamps[0] * double(fps);

    cout << "we have " << est_Pose_Timestamp.size() << " poses   " << endl;
    cout << "        " << Depth_Maps.size() << " depth map   " << endl;
    cout << "        " << Color_Imgs.size() << " color imgs! " << endl;

}

void
DataManager::readPoses(string file_path,
                       vector<double> &Pose_Timestamp,
                       std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &Poses_Twc,
                       bool set_first_pose_identity)
{
    cout << "est_pose_file " << file_path << endl;

    ifstream pose_ifs(file_path);
    while (!pose_ifs.eof())
    {

        std::string data_line_pose;
        std::string pose_time_stamp;
        Matrix4d trans;
        Matrix4d Tcw;
        Matrix3d Rot;
        Vector3d tra;
        Quaterniond q;
        std::getline(pose_ifs, data_line_pose);
        std::istringstream poseData(data_line_pose);
        poseData >> pose_time_stamp;
        poseData >> tra.x() >> tra.y() >> tra.z() >> q.x() >> q.y() >> q.z() >> q.w();
        Tcw.setIdentity();
        Tcw.block<3, 3>(0, 0) = q.toRotationMatrix();
        Tcw.block<3, 1>(0, 3) = tra;

        Poses_Twc.push_back(Tcw);

        double pose_time = atof(pose_time_stamp.c_str());
        Pose_Timestamp.push_back(pose_time);
    }


    if (set_first_pose_identity)
    {
        Eigen::Matrix4d trans = Poses_Twc[0];
        trans.block(0, 0, 3, 3).transposeInPlace();
        trans.block(0, 3, 3, 1) = trans.block(0, 0, 3, 3) * trans.block(0, 3, 3, 1);
        for (int i = 0; i < Poses_Twc[i].size(); ++i)
        {
            Poses_Twc[i] = Poses_Twc[i] * trans;
        }
    }
}

void
DataManager::get_and_align_poses()
{

    readPoses(Cfgparam.est_pose_file, est_Pose_Timestamp, est_Poses_Twc, true);
    cout << "gt_pose_file " << Cfgparam.gt_pose_file << endl;
    if (!Cfgparam.apply_scale)
        return;

    readPoses(Cfgparam.gt_pose_file, gt_Pose_Timestamp, gt_Poses_Twc, false);  //第一帧已经设置为原点
    cout << "est_pose_file " << Cfgparam.est_pose_file << endl;


    int cnter = 0;
    int win_size = 10;
    double curr_scale = 0;
    vector<double> v_scale;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> est_Poses_Twc_tmp;

    for (int i = 0; i < est_Pose_Timestamp.size(); ++i)
    {
        double curr_time = est_Pose_Timestamp[i];
        cout << "finding " << curr_time << endl;

        while (abs(gt_Pose_Timestamp[cnter] - curr_time) > 1.0 / 60.0)
        {
            cnter++;
        }

        curr_scale = gt_Poses_Twc[cnter].block(0, 3, 3, 1).norm() / est_Poses_Twc[i].block(0, 3, 3, 1).norm();
        cout << "curr_scale:" << curr_scale << endl;

        v_scale.push_back(curr_scale);

        if (v_scale.size() > win_size)
        {
            v_scale.erase(v_scale.begin());
        }

        double sum = 0;
        for (int j = 0; j < v_scale.size(); ++j)
            sum += v_scale[j];
        double mean_scale = sum / (double) v_scale.size();
        gt_Poses_Twc[cnter].block(0, 0, 3, 3) /= mean_scale;
        est_Poses_Twc_tmp.push_back(gt_Poses_Twc[cnter]);
        scale_gt_div_est.push_back(mean_scale);
    }

    gt_Poses_Twc.clear();
    gt_Poses_Twc = est_Poses_Twc_tmp;
}

bool
DataManager::grebFrame(cv::Mat &color_raw,
                       cv::Mat &depth,
                       cv::Mat &depth_gt,
                       Matrix4d &Twc,
                       double &curr_scale,
                       double &curr_time_stamp)
{

    string depth_path = Depth_Maps[depth_idx];
    depth = imread(depth_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

    depth_idx++;

    string s_depth_time = depth_path
        .substr(depth_path.find_last_of("/") + 1,
                depth_path.find_last_of(".") - depth_path.find_last_of("/") - 1);

    curr_time_stamp = atof(s_depth_time.c_str());

    double d_color_time = -1;
    string color_path;

    bool is_images = (Cfgparam.data_formate == "IMAGES");

    while (is_images && abs(d_color_time - curr_time_stamp) > 1.0 / 60.0)
    {
        color_path = Color_Imgs[color_idx];
        color_idx++;
        string s_color_time = color_path
            .substr(color_path.find_last_of("/") + 1,
                    color_path.find_last_of(".") - color_path.find_last_of("/") - 1);
        d_color_time = atof(s_color_time.c_str());
    }

    if (is_images)
        color_raw = imread(color_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    else
    {
//        while (abs(((double)color_idx/30.0) - curr_time_stamp))
        capture.set(CAP_PROP_POS_FRAMES, int(curr_time_stamp * 30.0));
        capture >> color_raw;
        color_idx++;
    }
//    cout<<"color_path:"<<color_path<<endl;
    if (color_raw.size() != Cfgparam.img_size)
    {
        cv::resize(color_raw, color_raw, Cfgparam.img_size, cv::INTER_AREA);
    }

    if (Cfgparam.apply_scale)
    {
        //    读取真值深度图
        string s_color_time = color_path
            .substr(color_path.find_last_of("/") + 1,
                    color_path.find_last_of(".") - color_path.find_last_of("/") - 1);
        string depth_gt_path = Cfgparam.depth_gt_path + "/" + s_color_time + ".png";
//    cout << "depth_gt_path = " << depth_gt_path << endl;
        depth_gt = imread(depth_gt_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        if (depth_gt.size() != Cfgparam.img_size)
        {
            cv::resize(depth_gt, depth_gt, Cfgparam.img_size, cv::INTER_AREA);
        }
    }


    double pose_time = est_Pose_Timestamp[pose_idx];
    while (abs(pose_time - curr_time_stamp) > 1.0 / 60.0)
    {
        pose_idx++;
        pose_time = est_Pose_Timestamp[pose_idx];
    }
    Twc = est_Poses_Twc[pose_idx];

    if (Cfgparam.apply_scale)
        curr_scale = scale_gt_div_est[pose_idx];

    return true;
}


