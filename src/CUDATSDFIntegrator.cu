//
// Created by will on 20-1-9.
//

#include "CUDATSDFIntegrator.h"

// CUDA kernel function to integrate a TSDF voxel volume given depth images and color images
__global__ void
IntegrateDepthMapKernel(float *d_cam_K, float *T_bc, float *d_depth, uchar3 *d_color,
                        float voxel_size, float truncation, int height, int width,
                        int grid_dim_x, int grid_dim_y, int grid_dim_z,
                        float grid_origin_x, float grid_origin_y, float grid_origin_z, Voxel *d_SDFBlocks)
{
    int pt_grid_z = threadIdx.x;
    int pt_grid_x = blockIdx.x;
    int pt_grid_y = blockIdx.y;

    float pt_x = grid_origin_x + (float) pt_grid_x * voxel_size;
    float pt_y = grid_origin_y + (float) pt_grid_y * voxel_size;
    float pt_z = grid_origin_z + (float) pt_grid_z * voxel_size;

    // Converter voxel center from grid voxel coordinates to real world coordinates

    // Converter world coordinates to current camera coordinates
    float tmp[3] = {0};
    tmp[0] = pt_x - T_bc[0 * 4 + 3];
    tmp[1] = pt_y - T_bc[1 * 4 + 3];
    tmp[2] = pt_z - T_bc[2 * 4 + 3];
    float P_b_x = T_bc[0 * 4 + 0] * tmp[0] + T_bc[1 * 4 + 0] * tmp[1] + T_bc[2 * 4 + 0] * tmp[2];
    float P_b_y = T_bc[0 * 4 + 1] * tmp[0] + T_bc[1 * 4 + 1] * tmp[1] + T_bc[2 * 4 + 1] * tmp[2];
    float P_b_z = T_bc[0 * 4 + 2] * tmp[0] + T_bc[1 * 4 + 2] * tmp[1] + T_bc[2 * 4 + 2] * tmp[2];

    if (P_b_z <= 0)
        return;

    // d_camK: fx, fy, cx, cy
    int pt_pix_x = roundf(d_cam_K[0] * (P_b_x / P_b_z) + d_cam_K[2]);
    int pt_pix_y = roundf(d_cam_K[1] * (P_b_y / P_b_z) + d_cam_K[3]);

    if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)  //当前网格在视野之外
        return;

    //printf("%d, %d\n", pt_pix_x, pt_pix_y);
    float depth_val = d_depth[pt_pix_y * width + pt_pix_x]; //横排的二维数组suoy y * w + x
    if (depth_val <= 0 || depth_val > 6)
        return;

    float diff = depth_val - P_b_z;  //diff是 估计的当前坐标系下的深度 减去 格子在当前帧位姿下的深度
    if (diff <= -truncation)
        return;

    int volume_idx = pt_grid_z * grid_dim_x * grid_dim_y +
        pt_grid_y * grid_dim_x +
        pt_grid_x;
//        int z = floor(i / (h_gridSize_x * h_gridSize_y));
//        int y = floor((i - (z * h_gridSize_x * h_gridSize_y)) / h_gridSize_x);
//        int x = i - (z * h_gridSize_x * h_gridSize_y) - (y * h_gridSize_y);

    // Integrate TSDF
    float dist = fmin(1.0f, diff / truncation);
    float weight_old = d_SDFBlocks[volume_idx].weight;
    float weight_new = weight_old + 1.0f;
    d_SDFBlocks[volume_idx].weight = weight_new;
    d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + dist) / weight_new;
//    __syncthreads();
    // Integrate Color
    uchar3 RGB = d_color[pt_pix_y * width + pt_pix_x];
    float3 cur_color = make_float3(RGB.x, RGB.y, RGB.z);
    float3 old_color = make_float3(d_SDFBlocks[volume_idx].color.x,
                                   d_SDFBlocks[volume_idx].color.y, d_SDFBlocks[volume_idx].color.z);
    float3 new_color;
    new_color.x = fmin(roundf((old_color.x * weight_old + cur_color.x) / weight_new), 255.0f);
    new_color.y = fmin(roundf((old_color.y * weight_old + cur_color.y) / weight_new), 255.0f);;
    new_color.z = fmin(roundf((old_color.z * weight_old + cur_color.z) / weight_new), 255.0f);;
    d_SDFBlocks[volume_idx].color = make_uchar3(new_color.x, new_color.y, new_color.z);
}

__global__ void
IntegrateDepthMapKernel_single(float *d_cam_K, float *T_bc, float *d_depth, uchar3 *d_color,
                               float voxel_size, float truncation, int height, int width,
                               int grid_dim_x, int grid_dim_y, int grid_dim_z,
                               float grid_origin_x, float grid_origin_y, float grid_origin_z, Voxel *d_SDFBlocks)
{
    const int pix_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pix_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pix_x >= width - 1 || pix_y >= height - 1 || pix_x <= 0 || pix_y <= 0)
        return;

    int pt_grid_x;
    int pt_grid_y;
    int pt_grid_z;
    float P_z_c_grid;
    int volume_idx;
    float P_z_c_est;

//    计算当前估计的像素在世界坐标系下的坐标
    {
        P_z_c_est = d_depth[pix_y * width + pix_x];

        if (P_z_c_est == 0)
            return;

        float P_x_c_est = P_z_c_est * (float(pix_x) - d_cam_K[2]) / (d_cam_K[0]);
        float P_y_c_est = P_z_c_est * (float(pix_y) - d_cam_K[3]) / (d_cam_K[1]);

        float tmp[3] = {P_x_c_est, P_y_c_est, P_z_c_est};
        tmp[0] = T_bc[0 * 4 + 0] * P_x_c_est + T_bc[0 * 4 + 1] * P_y_c_est + T_bc[0 * 4 + 2] * P_z_c_est;
        tmp[1] = T_bc[1 * 4 + 0] * P_x_c_est + T_bc[1 * 4 + 1] * P_y_c_est + T_bc[1 * 4 + 2] * P_z_c_est;
        tmp[2] = T_bc[2 * 4 + 0] * P_x_c_est + T_bc[2 * 4 + 1] * P_y_c_est + T_bc[2 * 4 + 2] * P_z_c_est;
        float P_x_b = tmp[0] + T_bc[0 * 4 + 3];
        float P_y_b = tmp[1] + T_bc[1 * 4 + 3];
        float P_z_b = tmp[2] + T_bc[2 * 4 + 3];

//    有了世界坐标系下的坐标,寻找其属于哪个box,超出格子的要直接丢掉
        if (P_x_b < grid_origin_x || P_x_b > grid_origin_x + voxel_size * grid_dim_x)
            return;
        pt_grid_x = int((P_x_b - grid_origin_x) / voxel_size);

        if (P_y_b < grid_origin_y || P_y_b > grid_origin_y + voxel_size * grid_dim_y)
            return;
        pt_grid_y = int((P_y_b - grid_origin_y) / voxel_size);

        if (P_z_b < grid_origin_z || P_z_b > grid_origin_z + voxel_size * grid_dim_z)
            return;
        pt_grid_z = int((P_z_b - grid_origin_z) / voxel_size);

        volume_idx = pt_grid_z * grid_dim_x * grid_dim_y +
            pt_grid_y * grid_dim_x +
            pt_grid_x;

        if (volume_idx >= grid_dim_x * grid_dim_y * grid_dim_z ||
            volume_idx < 0
            )
        {
            return;
        }
    }


//    投回去,计算差值
    {
        float pt_x = grid_origin_x + (float) pt_grid_x * voxel_size;
        float pt_y = grid_origin_y + (float) pt_grid_y * voxel_size;
        float pt_z = grid_origin_z + (float) pt_grid_z * voxel_size;

        // Converter world coordinates to current camera coordinates
        float tmp_re[3] = {0};
        tmp_re[0] = pt_x - T_bc[0 * 4 + 3];
        tmp_re[1] = pt_y - T_bc[1 * 4 + 3];
        tmp_re[2] = pt_z - T_bc[2 * 4 + 3];

        float P_x_c_grid = T_bc[0 * 4 + 0] * tmp_re[0] + T_bc[1 * 4 + 0] * tmp_re[1] + T_bc[2 * 4 + 0] * tmp_re[2];
        float P_y_c_grid = T_bc[0 * 4 + 1] * tmp_re[0] + T_bc[1 * 4 + 1] * tmp_re[1] + T_bc[2 * 4 + 1] * tmp_re[2];
        P_z_c_grid = T_bc[0 * 4 + 2] * tmp_re[0] + T_bc[1 * 4 + 2] * tmp_re[1] + T_bc[2 * 4 + 2] * tmp_re[2];
    }

//    计算了差值,更新sdf
    {
        float diff = P_z_c_est - P_z_c_grid;  //diff是 估计的当前坐标系下的深度 减去 格子在当前帧位姿下的深度
        if (diff <= -truncation)
            return;

        // Integrate TSDF
        float dist = fmin(1.0f, diff / truncation);
        float weight_old = d_SDFBlocks[volume_idx].weight;
        float weight_new = weight_old + 1.0f;
        d_SDFBlocks[volume_idx].weight = weight_new;
        d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + dist) / weight_new;

        // Integrate Color
        uchar3 RGB = d_color[pix_y * width + pix_x];
        float3 cur_color = make_float3(RGB.x, RGB.y, RGB.z);
        float3 old_color = make_float3(d_SDFBlocks[volume_idx].color.x,
                                       d_SDFBlocks[volume_idx].color.y, d_SDFBlocks[volume_idx].color.z);
        float3 new_color;
        new_color.x = fmin(roundf((old_color.x * weight_old + cur_color.x) / weight_new), 255.0f);
        new_color.y = fmin(roundf((old_color.y * weight_old + cur_color.y) / weight_new), 255.0f);;
        new_color.z = fmin(roundf((old_color.z * weight_old + cur_color.z) / weight_new), 255.0f);;
        d_SDFBlocks[volume_idx].color = make_uchar3(new_color.x, new_color.y, new_color.z);
    }
}

__global__ void
IntegrateDepthMapKernel_multi(float *d_cam_K,
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
                              float grid_origin_x,
                              float grid_origin_y,
                              float grid_origin_z,
                              Voxel *d_SDFBlocks,
                              float *exceed_num
                              )
{

//    const int pix_x = threadIdx.x + blockIdx.x * blockDim.x;
//    const int pix_y = threadIdx.y + blockIdx.y * blockDim.y;
//
//    if (pix_x >= width - 1 || pix_y >= height - 1 || pix_x <= 0 || pix_y <= 0)
//        return;

    const int pix_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pix_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pix_x >= width || pix_y >= height)
        return;


    int pt_grid_x;
    int pt_grid_y;
    int pt_grid_z;
    float P_z_c_grid;
    int volume_idx;
    float P_z_c_est;

    float threshold;
    float ang;
    float P_x_c_norm;
    float P_y_c_norm;

//    计算当前估计的像素在世界坐标系下的坐标
    {
        P_z_c_est = d_depth[pix_y * width + pix_x];

        if (P_z_c_est <= 0 || P_z_c_est >= 6.0)
            return;

        P_x_c_norm = (float(pix_x) - d_cam_K[2]) / (d_cam_K[0]);  //[fx,fy,cx,cy]
        P_y_c_norm = (float(pix_y) - d_cam_K[3]) / (d_cam_K[1]);
        float P_x_c_est = P_z_c_est * P_x_c_norm;
        float P_y_c_est = P_z_c_est * P_y_c_norm;

        float P_x_b = T_bc[0 * 4 + 0] * P_x_c_est + T_bc[0 * 4 + 1] * P_y_c_est + T_bc[0 * 4 + 2] * P_z_c_est+ T_bc[0 * 4 + 3];
        float P_y_b = T_bc[1 * 4 + 0] * P_x_c_est + T_bc[1 * 4 + 1] * P_y_c_est + T_bc[1 * 4 + 2] * P_z_c_est+ T_bc[1 * 4 + 3];
        float P_z_b = T_bc[2 * 4 + 0] * P_x_c_est + T_bc[2 * 4 + 1] * P_y_c_est + T_bc[2 * 4 + 2] * P_z_c_est+ T_bc[2 * 4 + 3];

//    有了世界坐标系下的坐标,寻找其属于哪个box,超出格子的要直接丢掉
        if (P_x_b < grid_origin_x || P_x_b > grid_origin_x + voxelSize_x * (float) (grid_dim_x-1))
        {
            atomicAdd(&exceed_num[0], 1.0f);
            return;
        }

        if (P_y_b < grid_origin_y || P_y_b > grid_origin_y + voxelSize_y * (float) (grid_dim_y-1))
        {
            atomicAdd(&exceed_num[1], 1.0f);
            return;
        }

        if (P_z_b < grid_origin_z || P_z_b > grid_origin_z + voxelSize_z * (float) (grid_dim_z-1))
        {
            atomicAdd(&exceed_num[2], 1.0f);
            return;
        }

        __syncthreads();

        pt_grid_x = int((P_x_b - grid_origin_x) / voxelSize_x);
        pt_grid_y = int((P_y_b - grid_origin_y) / voxelSize_y);
        pt_grid_z = int((P_z_b - grid_origin_z) / voxelSize_z);

        volume_idx = pt_grid_z * grid_dim_x * grid_dim_y +
            pt_grid_y * grid_dim_x +
            pt_grid_x;

//        if(volume_idx>grid_dim_x*grid_dim_y*grid_dim_z)
//        {
//            atomicAdd(&exceed_num[2], 1);
//            return;
//        }

    }

//    threshold =  ((- d_cam_K[2]) /(d_cam_K[0]))*((- d_cam_K[2]) /(d_cam_K[0])) + (  (- d_cam_K[3]) / (d_cam_K[1]) )*(  (- d_cam_K[3]) / (d_cam_K[1]) );
//    ang = sqrt( threshold / ((P_x_c_norm*P_x_c_norm ) + (  P_y_c_norm*P_y_c_norm)) );

//    投回去,计算差值
    {
        float pt_x = grid_origin_x + (float) pt_grid_x * voxelSize_x;
        float pt_y = grid_origin_y + (float) pt_grid_y * voxelSize_y;
        float pt_z = grid_origin_z + (float) pt_grid_z * voxelSize_z;

        // Converter world coordinates to current camera coordinates
        float tmp_re[3] = {0};
        tmp_re[0] = pt_x - T_bc[0 * 4 + 3];
        tmp_re[1] = pt_y - T_bc[1 * 4 + 3];
        tmp_re[2] = pt_z - T_bc[2 * 4 + 3];

//        float P_x_c_grid = T_bc[0 * 4 + 0] * tmp_re[0] + T_bc[1 * 4 + 0] * tmp_re[1] + T_bc[2 * 4 + 0] * tmp_re[2];
//        float P_y_c_grid = T_bc[0 * 4 + 1] * tmp_re[0] + T_bc[1 * 4 + 1] * tmp_re[1] + T_bc[2 * 4 + 1] * tmp_re[2];
        P_z_c_grid = T_bc[0 * 4 + 2] * tmp_re[0] + T_bc[1 * 4 + 2] * tmp_re[1] + T_bc[2 * 4 + 2] * tmp_re[2];
    }

//    计算了差值,更新sdf
    {
        float diff = P_z_c_est - P_z_c_grid;  //diff是 估计的当前坐标系下的深度 减去 格子在当前帧位姿下的深度

//        if (diff <= -truncation)
//            return;

        // Integrate TSDF
        float dist = fmin(1.0f, diff / truncation);
        float weight_old = d_SDFBlocks[volume_idx].weight;
        float weight_new = weight_old + 1.0f;
//        float weight_new = weight_old + ang;
        d_SDFBlocks[volume_idx].weight = weight_new;
        d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + dist) / weight_new;

        // Integrate Color
        uchar3 RGB = d_color[pix_y * width + pix_x];
        float3 cur_color = make_float3(RGB.x, RGB.y, RGB.z);
        float3 old_color = make_float3(d_SDFBlocks[volume_idx].color.x,
                                       d_SDFBlocks[volume_idx].color.y,
                                       d_SDFBlocks[volume_idx].color.z);
        float3 new_color;
        new_color.x = fmin(roundf((old_color.x * weight_old + cur_color.x) / weight_new), 255.0f);
        new_color.y = fmin(roundf((old_color.y * weight_old + cur_color.y) / weight_new), 255.0f);
        new_color.z = fmin(roundf((old_color.z * weight_old + cur_color.z) / weight_new), 255.0f);
        d_SDFBlocks[volume_idx].color = make_uchar3(new_color.x, new_color.y, new_color.z);
    }
}

__global__ void
calc_voxel_field_kernel(const int width,
                        const int height,
                        const int hist_num,
                        const float element,
                        const float3 mean_pts,
                        float3 *d_curr_p3d,
                        int3 *d_hist
)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if ( d_curr_p3d[idx].z > 0.0f )
    {
        int3 hist_id;

        hist_id.x = abs(int((d_curr_p3d[idx].x - mean_pts.x) / element));
        hist_id.y = abs(int((d_curr_p3d[idx].y - mean_pts.y) / element));
        hist_id.z = abs(int((d_curr_p3d[idx].z - mean_pts.z) / element));

        hist_id.x = hist_id.x >= hist_num ? hist_num-1:hist_id.x;
        hist_id.y = hist_id.y >= hist_num ? hist_num-1:hist_id.y;
        hist_id.z = hist_id.z >= hist_num ? hist_num-1:hist_id.z;

        atomicAdd(&d_hist[hist_id.x].x, 1);
        atomicAdd(&d_hist[hist_id.y].y, 1);
        atomicAdd(&d_hist[hist_id.z].z, 1);
    }

}

void
calc_voxel_field(const int width,
                 const int height,
                 const int hist_num,
                 const float element,
                 const float3 mean_pts,
                 float3 *d_curr_p3d,
                 int3 *d_hist)
{
    dim3 image_block;
    dim3 image_grid;
    image_block.x = 16;
    image_block.y = 16;
    image_grid.x = (width + image_block.x - 1) / image_block.x;
    image_grid.y = (height + image_block.y - 1) / image_block.y;

    calc_voxel_field_kernel<<<image_grid, image_block>>>(width,
                                                         height,
                                                         hist_num,
                                                         element,
                                                         mean_pts,
                                                         d_curr_p3d,
                                                         d_hist);

}

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
                      float grid_origin_x,
                      float grid_origin_y,
                      float grid_origin_z,
                      Voxel *d_SDFBlocks,
                      float *d_exceed_num)
{

#if 1
    dim3 image_block;
    dim3 image_grid;
    image_block.x = 16;
    image_block.y = 16;
    image_grid.x = (width + image_block.x - 1) / image_block.x;
    image_grid.y = (height + image_block.y - 1) / image_block.y;

    float h_exceed_num[3] = {0, 0, 0};
    checkCudaErrors(cudaMemcpy(d_exceed_num, h_exceed_num, 3 * sizeof(float), cudaMemcpyHostToDevice));
    IntegrateDepthMapKernel_multi<<< image_grid, image_block >>>(d_cam_K,
                                                     T_bc,
                                                     d_depth,
                                                     d_color,
                                                     voxelSize_x,
                                                     voxelSize_y,
                                                     voxelSize_z,
                                                     truncation,
                                                     height,
                                                     width,
                                                     grid_dim_x,
                                                     grid_dim_y,
                                                     grid_dim_z,
                                                     grid_origin_x,
                                                     grid_origin_y,
                                                     grid_origin_z,
                                                     d_SDFBlocks,
                                                     d_exceed_num
    );


#else
    dim3 blockSize;
    blockSize.x = grid_dim_x;
    blockSize.y = grid_dim_y;

    dim3 gridSize(grid_dim_z);

    std::cout << "Launch Kernel...:\n grid : [" << gridSize.x << "," << gridSize.y << "] \n thread: [" << blockSize.x
              << ", " << blockSize.y << "]" << std::endl;

    IntegrateDepthMapKernel <<< blockSize, gridSize >>>(d_cam_K,
                                                        T_bc,
                                                        d_depth,
                                                        d_color,
                                                        voxel_size,
                                                        truncation,
                                                        height,
                                                        width,
                                                        grid_dim_x,
                                                        grid_dim_y,
                                                        grid_dim_z,
                                                        grid_origin_x,
                                                        grid_origin_y,
                                                        grid_origin_z,
                                                        d_SDFBlocks);
#endif
    std::cout << "finished..." << std::endl;

    //cudaError_t status = cudaGetLastError();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}