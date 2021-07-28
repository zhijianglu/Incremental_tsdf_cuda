cmake_minimum_required(VERSION 3.10)
project(INC_TSDF CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -pthread -lrt -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
set(PCL_DIR /usr/local/include/pcl-1.9)

find_package(PCL 1.9 REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CUDA_NVCC_FLAGS -arch=sm_50;-std=c++11;-lm;)
ADD_DEFINITIONS(-fpermissive)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math )
set(__cuda_arch_bin "5.2")
set(__cuda_arch_ptx "5.2")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

# solve the problem: nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "")

cuda_add_executable(inc_tsdf app/main_whole_scene.cpp
        src/Reader.cpp
        src/CUDATSDFIntegrator.cpp
        src/DataManager.cpp
        src/parameters.cpp
        src/CUDATSDFIntegrator.cu
        )
target_link_libraries(inc_tsdf ${CUDA_LIBRARIES} ${OpenCV_LIBS}  ${Boost_LIBRARIES} ${PCL_LIBRARIES})




cuda_add_executable(code_test app/main_code_test.cpp
        src/Reader.cpp
        src/pcl_cuda.cu
        src/CUDATSDFIntegrator.cpp
        src/DataManager.cpp
        src/parameters.cpp
        src/CUDATSDFIntegrator.cu
        )
target_link_libraries(code_test  ${CUDA_LIBRARIES} ${OpenCV_LIBS}  ${Boost_LIBRARIES} ${PCL_LIBRARIES})

