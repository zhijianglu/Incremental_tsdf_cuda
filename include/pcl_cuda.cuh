
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>

#include <queue>
#include <map>
#include <math.h>
#include <thread>
#include <mutex>
#include <cuda.h>
#include <opencv2/opencv.hpp>

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

typedef std::numeric_limits<float> Info_f;
typedef std::numeric_limits<uchar> Info_uc;
float const NAN_f = Info_f::quiet_NaN();
float const NAN_uc = Info_uc::quiet_NaN();

__global__ void
pcl_cuda_test(
    int width,
    int height,
    pcl::PointXYZRGB* pts);
