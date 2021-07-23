//
// Created by will on 20-1-9.
//

#include "CUDATSDFIntegrator.h"
#include "parameters.h"
#include <numeric>

void
IntegrateDepthMapCUDA(float *d_cam_K,
                      float *T_bc,
                      float *d_depth,
                      uchar3 *d_color,
                      float voxelSize_x,
                      float voxelSize_y,
                      float voxelSize_z,
                      float truncation,
                      int height,
                      int width,
                      int grid_dim_x,
                      int grid_dim_y,
                      int grid_dim_z,
                      float gird_origin_x,
                      float gird_origin_y,
                      float gird_origin_z,
                      Voxel *d_SDFBlocks,
                      float *d_exceed_num);

void
calc_voxel_field(const int width,
                 const int height,
                 const int hist_num,
                 const float element,
                 const float3 mean_pts,
                 float3 *d_curr_p3d,
                 int3 *d_hist);

void
deIntegrateDepthMapCUDA();

CUDATSDFIntegrator::CUDATSDFIntegrator()
{
    // Camera Intrinsics
    h_camK[0] = (float) Cfgparam.fx;
    h_camK[1] = (float) Cfgparam.fy;
    h_camK[2] = (float) Cfgparam.cx;
    h_camK[3] = (float) Cfgparam.cy;

    std::cout << "[fx,fy,cx,cy]: " << h_camK[0] << "," << h_camK[1] << "," << h_camK[2] << "," << h_camK[3]
              << std::endl;

    h_width = Cfgparam.img_size.width;
    h_height = Cfgparam.img_size.height;
//    h_voxelSize = Cfgparam.VoxelSize;
    TruncationScale = Cfgparam.TruncationScale;

    h_gridSize_x = Cfgparam.GridSize_x;
    h_gridSize_y = Cfgparam.GridSize_y;
    h_gridSize_z = Cfgparam.GridSize_z;

//    初始化计算voxel field
    h_curr_p3d = new float3[h_width * h_height];
    h_hist = new int3[hist_num];
    checkCudaErrors(cudaMalloc(&d_curr_p3d, h_height * h_width * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_hist, hist_num * sizeof(int3)));

    exceed_thd = make_float3(
        float(h_width * h_height) * (0.03f),
        float(h_width * h_height) * (0.03f),
        float(h_width * h_height) * (0.02f)
    );
}

void
CUDATSDFIntegrator::Initialize(cv::Mat &depth_img, cv::Mat &depth_gt_img, Eigen::Matrix4d &e_Twc,
                               float *depth, double scale_gt_div_est)
{
    int width = Cfgparam.img_size.width;
    int height = Cfgparam.img_size.height;
    memset(depth, 0.0f, width * height); //清零
    int start_line = 0;
    vector<Eigen::Vector3d> pt;
    int tol_valid_num = 0;
    float mean_x = 0;
    float mean_y = 0;
    float mean_z = 0;

    float sum_x = 0;
    float sum_y = 0;
    float sum_z = 0;

    for (int r = start_line; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int idx = r * width + c;

            float p3d_z = (float) ((depth_img.at<char16_t>(r, c)) / Cfgparam.depth_result_factor);

            if (p3d_z < 0.2 || p3d_z > 6.0 || isinf(p3d_z) || isnan(p3d_z))
            {
                depth[r * width + c] = 0.0;
                h_curr_p3d[idx].x = 0.0f;
                h_curr_p3d[idx].y = 0.0f;
                h_curr_p3d[idx].z = 0.0f;
            }
            else
            {
                float p3d_x = p3d_z * ((float) c - (float) Cfgparam.cx) / (float) Cfgparam.fx;
                float p3d_y = p3d_z * ((float) r - (float) Cfgparam.cy) / (float) Cfgparam.fy;

                h_curr_p3d[idx].x = p3d_x;
                h_curr_p3d[idx].y = p3d_y;
                h_curr_p3d[idx].z = p3d_z;

                sum_x += p3d_x;
                sum_y += p3d_y;
                sum_z += p3d_z;

                depth[idx] = p3d_z;
                tol_valid_num++;
            }
        }
    }
    mean_x = sum_x / (float) tol_valid_num;
    mean_y = sum_y / (float) tol_valid_num;
    mean_z = sum_z / (float) tol_valid_num;
    cudaMemcpy(d_curr_p3d, h_curr_p3d, h_width * h_height * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_num * sizeof(int3));

    float MAX_RANGE = 10.0f;
    float element = MAX_RANGE / (float) (hist_num);
    cout << "element:" << element << endl;
    float3 mean_pts = make_float3(mean_x, mean_y, mean_z);
    calc_voxel_field(h_width, h_height, hist_num, element, mean_pts, d_curr_p3d, d_hist);
    cudaMemcpy(h_hist, d_hist, hist_num * sizeof(int3), cudaMemcpyDeviceToHost);
    int3 th = make_int3(Cfgparam.voxelField_rate_x * (double) tol_valid_num,
                        Cfgparam.voxelField_rate_y * (double) tol_valid_num,
                        Cfgparam.voxelField_rate_z * (double) tol_valid_num);
    int3 hist_tgt = make_int3(-1, -1, -1);
    int3 hist_sum = make_int3(0, 0, 0);

//    for (int j = 0; j < hist_num; ++j){
//        cout << " " << h_hist[j].x << " " << h_hist[j].y << " " << h_hist[j].z << endl;
//    }

    for (int i = 0; i < hist_num; ++i)
    {
        hist_sum.x += h_hist[i].x;
        if (hist_sum.x > th.x && hist_tgt.x < 0)
            hist_tgt.x = i;

        hist_sum.y += h_hist[i].y;
        if (hist_sum.y > th.y && hist_tgt.y < 0)
            hist_tgt.y = i;

        hist_sum.z += h_hist[i].z;
        if (hist_sum.z > th.z && hist_tgt.z < 0)
            hist_tgt.z = i;

        if (hist_tgt.x > 0 && hist_tgt.y > 0 && hist_tgt.z > 0)
            break;
    }

    int sum_test_x = 0;
    int sum_test_y = 0;
    int sum_test_z = 0;

    float3 scale_r;  //求解出范围
    scale_r.x = (1.1f) * element * (float) hist_tgt.x;
    scale_r.y = (1.1f) * element * (float) hist_tgt.y;
    scale_r.z = (1.2f) * element * (float) hist_tgt.z;

    h_grid_origin_x = (float) mean_x - scale_r.x;
    h_grid_end_x = (float) mean_x + scale_r.x;
    h_voxelSize_x = (h_grid_end_x - h_grid_origin_x) / (float) h_gridSize_x;

    h_grid_origin_z = (float) mean_z - scale_r.z;
    h_grid_end_z = (float) mean_z + scale_r.z;
    h_voxelSize_z = (h_grid_end_z - h_grid_origin_z) / (float) h_gridSize_z;

//    因为是朝前走的,y在负空间内拓展
    h_voxelSize_y = h_voxelSize_x;
    h_grid_end_y = (float) mean_y + element * (float) hist_tgt.y;
    h_grid_origin_y = h_grid_end_y - h_voxelSize_y * (float) h_gridSize_y;

    h_truncation = TruncationScale * h_voxelSize_z;

    for (int i = 0; i < h_width * h_height; ++i)
    {
        if (h_curr_p3d[i].z == 0.0f)
            continue;

        float pt_x = h_curr_p3d[i].x;
        float pt_y = h_curr_p3d[i].y;
        float pt_z = h_curr_p3d[i].z;

        if ((pt_z >= h_grid_origin_z) && (pt_z <= h_grid_end_z))
            sum_test_z++;

        if ((pt_y >= h_grid_origin_y) && (pt_y <= h_grid_end_y))
            sum_test_y++;

        if ((pt_x >= h_grid_origin_x) && (pt_x <= h_grid_end_x))
            sum_test_x++;
    }


    std::cout << "  x percent: " << (double) sum_test_x / (double) tol_valid_num << std::endl;
    std::cout << "  y percent: " << (double) sum_test_y / (double) tol_valid_num << std::endl;
    std::cout << "  z percent: " << (double) sum_test_z / (double) tol_valid_num << std::endl;

    std::cout << "  x range: [" << h_grid_origin_x << "--" << h_grid_end_x << "] " << std::endl;
    std::cout << "  y range: [" << h_grid_origin_y << "--" << h_grid_end_y << "] " << std::endl;
    std::cout << "  z range: [" << h_grid_origin_z << "--" << h_grid_end_z << "] " << std::endl;

    std::cout << "TruncationScale: " << TruncationScale << std::endl;
    std::cout << " h_voxelSize_x: " << h_voxelSize_x << std::endl;
    std::cout << " h_voxelSize_y: " << h_voxelSize_y << std::endl;
    std::cout << " h_voxelSize_z: " << h_voxelSize_z << std::endl;
    std::cout << "  GridSize: " << h_gridSize_x << std::endl;
    std::cout << "Initialize TSDF ..." << std::endl;

    if (!is_initialized)
    {
        h_SDFBlocks = new Voxel[h_gridSize_x * h_gridSize_y * h_gridSize_z];

        checkCudaErrors(cudaMalloc(&d_camK, 4 * sizeof(float)));

        checkCudaErrors(cudaMemcpy(d_camK, h_camK, 4 * sizeof(float), cudaMemcpyHostToDevice));
        // TSDF model
        checkCudaErrors(cudaMalloc(&d_SDFBlocks, h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel)));
        // depth data
        checkCudaErrors(cudaMalloc(&d_depth, h_height * h_width * sizeof(float)));
        // color data
        checkCudaErrors(cudaMalloc(&d_color, h_height * h_width * sizeof(uchar3)));
        // pose in base coordinates
        checkCudaErrors(cudaMalloc(&T_bc, 4 * 4 * sizeof(float)));

        checkCudaErrors(cudaMalloc(&exceed_num, 3 * sizeof(float)));

        is_initialized = true;
    }

    if (need_reset)
    {
        delete[]h_SDFBlocks;
        h_SDFBlocks = new Voxel[h_gridSize_x * h_gridSize_y * h_gridSize_z];
        checkCudaErrors(cudaMemcpy(d_SDFBlocks, h_SDFBlocks,
                                   h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyHostToDevice));

        need_reset = false;
    }
}

Voxel *
CUDATSDFIntegrator::point(int pt_grid_x, int pt_grid_y, int pt_grid_z)
{

    int volume_idx = pt_grid_z * h_gridSize_x * h_gridSize_y +
        pt_grid_y * h_gridSize_x +
        pt_grid_x;
    return &h_SDFBlocks[volume_idx];
}

// Integrate depth and color into TSDF model
void
CUDATSDFIntegrator::integrate(float *depth_cpu_data, uchar *color_cpu_data, float *T_bc_)
{
    //std::cout << "Fusing color image and depth" << std::endl;

    // copy data to gpu
    TicToc timer;
    checkCudaErrors(cudaMemcpy(d_depth, depth_cpu_data, h_height * h_width * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_color, color_cpu_data, 3 * h_height * h_width * sizeof(uchar), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(T_bc, T_bc_, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice));

    cout << "load GPU memory cost:" << timer.toc() << " ms" << endl;

    // Integrate function
    IntegrateDepthMapCUDA(d_camK,
                          T_bc,
                          d_depth,
                          d_color,
                          h_voxelSize_x,
                          h_voxelSize_y,
                          h_voxelSize_z,
                          h_truncation,
                          h_height,
                          h_width,
                          h_gridSize_x,
                          h_gridSize_y,
                          h_gridSize_z,
                          h_grid_origin_x,
                          h_grid_origin_y,
                          h_grid_origin_z,
                          d_SDFBlocks,
                          exceed_num
    );
    cout << "fusion cost:" << timer.toc() << " ms" << endl;

    checkCudaErrors(cudaMemcpy(h_exceed_num, exceed_num,
                               3 * sizeof(float), cudaMemcpyDeviceToHost));

    if (h_exceed_num[0] > exceed_thd.x || h_exceed_num[1] > exceed_thd.y || h_exceed_num[2] > exceed_thd.z)
    {
        cout << "h_exceed_num:  " << h_exceed_num[0] << " " << h_exceed_num[1] << " " << h_exceed_num[2] << endl;
        cout << "thd_exceed_num:" << exceed_thd.x << " " << exceed_thd.y << " " << exceed_thd.z << endl;
        need_reset = true;
    }


    FrameId++;
}

// deIntegrate depth and color from TSDF model
void
CUDATSDFIntegrator::deIntegrate(float *depth_cpu_data, uchar3 *color_cpu_data, float *pose)
{

}

// Compute surface points from TSDF voxel grid and save points to point cloud file
void
CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud(float tsdf_thresh, float weight_thresh, Eigen::Matrix4d Twc)
{

    checkCudaErrors(cudaMemcpy(h_SDFBlocks, d_SDFBlocks,
                               h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyDeviceToHost));

    pcl::PointCloud<pcl::PointXYZRGB> curr_pointcloud;
    curr_scene->clear();

    for (int i = 0; i < h_gridSize_x * h_gridSize_y * h_gridSize_z; i++)
    {
        if (std::abs(h_SDFBlocks[i].sdf) < tsdf_thresh && h_SDFBlocks[i].weight > weight_thresh)
        {
            // Compute voxel indices in int for higher positive number range
//            这里的xyz是指格子的排列坐标
            int z = floor(i / (h_gridSize_x * h_gridSize_y));
            int y = floor((i - (z * h_gridSize_x * h_gridSize_y)) / h_gridSize_x);
            int x = i - (z * h_gridSize_x * h_gridSize_y) - y * h_gridSize_x;

//            这里的pt_base xyz指的是在基坐标下的绝对坐标
            float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize_x;
            float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize_y;
            float pt_base_z = h_grid_origin_z + (float) z * h_voxelSize_z;

            pcl::PointXYZRGB point;
            point.x = pt_base_x;
            point.y = pt_base_y;
            point.z = pt_base_z;

            point.r = h_SDFBlocks[i].color.x;
            point.g = h_SDFBlocks[i].color.y;
            point.b = h_SDFBlocks[i].color.z;
            curr_scene->push_back(point);
        }
    }

    need_reset = true;
    pcl::transformPointCloud(*curr_scene, *curr_scene, Twc);
}

void
CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud_1(float tsdf_thresh, float weight_thresh, Eigen::Matrix4d Twc)
{

    checkCudaErrors(cudaMemcpy(h_SDFBlocks, d_SDFBlocks,
                               h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyDeviceToHost));

    pcl::PointCloud<pcl::PointXYZRGB> curr_pointcloud;
    curr_scene->clear();


    for (int y = 0; y < h_gridSize_y; ++y)
    {
        for (int x = 0; x < h_gridSize_x; ++x)
        {
            Voxel *best_voxel = nullptr;
            float sum_z = 0;
            float sum_weight = 0;
            int best_z = -1;
            for (int z = 0; z < h_gridSize_z; ++z)
            {
                Voxel *curr_voxel = point(x, y, z);

                if (std::abs(curr_voxel->sdf) < tsdf_thresh && curr_voxel->weight > weight_thresh)
                {
                    if (best_voxel == nullptr)
                    {
                        best_voxel = curr_voxel;
                        best_z = z;
                    }
                    else
                    {
                        sum_weight += curr_voxel->weight;
                        sum_z += curr_voxel->weight * (float) z;

                        if (curr_voxel->weight > best_voxel->weight && curr_voxel->sdf < best_voxel->sdf)
                        {
                            best_voxel = curr_voxel;
                            best_z = z;
                        }
                    }
                }
            }

            if (best_z != -1)
            {
                float best_z_ = sum_z / sum_weight;
                float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize_x;
                float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize_y;
                float pt_base_z = h_grid_origin_z + (float) best_z_ * h_voxelSize_z;

                pcl::PointXYZRGB point;
                point.x = pt_base_x;
                point.y = pt_base_y;
                point.z = pt_base_z;

                point.r = best_voxel->color.z;
                point.g = best_voxel->color.y;
                point.b = best_voxel->color.x;
                curr_scene->push_back(point);
            }
        }
    }
    pcl::transformPointCloud(*curr_scene, *curr_scene, Twc);
}

// Default deconstructor
CUDATSDFIntegrator::~CUDATSDFIntegrator()
{
    free(h_SDFBlocks);
    checkCudaErrors(cudaFree(d_camK));
    checkCudaErrors(cudaFree(d_SDFBlocks));
    checkCudaErrors(cudaFree(d_depth));
    checkCudaErrors(cudaFree(d_color));
}
