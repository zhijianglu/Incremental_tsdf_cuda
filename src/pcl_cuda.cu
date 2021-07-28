
#include "../include/pcl_cuda.cuh"

__global__ void
pcl_cuda_test(
    int width,
    int height,
    pcl::PointXYZRGB* pts)
{
    const int pix_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pix_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pix_x >= width || pix_y >= height)
        return;

    int idx = pix_x * width + pix_y;

    if (pix_x > width / 2)
    {
        pts[idx].x = (float) pix_x;
        pts[idx].y = (float) pix_y;
    }
    else
    {
        pts[idx].x = NAN;
    }
}

void
pcl_cuda_test_cpu(
    int width,
    int height,
    pcl::PointXYZRGB* pts
){

    dim3 image_block;
    dim3 image_grid;
    image_block.x = 16;
    image_block.y = 16;
    image_grid.x = (width + image_block.x - 1) / image_block.x;
    image_grid.y = (height + image_block.y - 1) / image_block.y;

    pcl_cuda_test<<< image_grid, image_block >>>(width,
                                                 height,
                                                 pts
    );

}
