//
// Created by will on 20-1-9.
//

#ifndef TSDF_CUDATSDFINTEGRATOR_H
#define TSDF_CUDATSDFINTEGRATOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <pcl/gpu/containers/device_array.h>

#include "Utils.h"
#include <math.h>
#include "Voxel.h"

class CUDATSDFIntegrator
{
public:

    CUDATSDFIntegrator();
    void Initialize(cv::Mat &depth_img, cv::Mat &depth_gt_img,Eigen::Matrix4d& e_Twc,
                    float *depth, double scale_gt_div_est);

    virtual ~CUDATSDFIntegrator();
    bool is_initialized = false;
    bool need_reset = false;

public:

    void integrate(float* depth_cpu_data, uchar* color_cpu_data, float* T_bc_);
    Voxel* point(int pt_grid_x, int pt_grid_y, int  pt_grid_z);

    void deIntegrate(float* depth_cpu_data, uchar3* color_cpu_data, float* pose);

    void SaveVoxelGrid2SurfacePointCloud(float tsdf_thresh, float weight_thresh,Eigen::Matrix4d Twc);
    void SaveVoxelGrid2SurfacePointCloud_1(float tsdf_thresh, float weight_thresh,float* Twc);
    int FrameId = 0;
    float h_exceed_num[3] = {0};

private:

//host    
    // fx,fy,cx,cy
    float h_camK[4];
    float3* h_curr_p3d;
    float3* d_curr_p3d;
    int3* h_hist;
    int3* d_hist;
    float3 exceed_thd;
    int hist_num = 1000;

    // Image resolution
    int h_width;

    double depth_mean = 1;
    double depth_variance = 0.2;

    int h_height;

    // VoxelSize
//    float h_voxelSize;

    float h_voxelSize_x;
    float h_voxelSize_y;
    float h_voxelSize_z;

    // TruncationScale
    float h_truncation;
    float TruncationScale;

    // Grid size
    int h_gridSize_x;
    int h_gridSize_y;
    int h_gridSize_z;

    // Location of voxel grid origin in base frame coordinates
    float h_grid_origin_x;
    float h_grid_origin_y;
    float h_grid_origin_z;

    float h_grid_end_x;
    float h_grid_end_y;
    float h_grid_end_z;

    // TSDF model
    Voxel* h_SDFBlocks;
    
//host


//device   
    float* d_camK;
    
    float* d_depth;

    float* T_bc;
    float* T_wb;
    float* exceed_num;

    pcl::gpu::DeviceArray<pcl::PointXYZRGB> d_cloud;

    uchar3* d_color;

    Voxel* d_SDFBlocks;
//device 

};

#endif //TSDF_CUDATSDFINTEGRATOR_H
