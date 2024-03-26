cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(nvdsparsebbox_mmyolo LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -shared -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
set(TensorRT_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu" CACHE STRING "TensorRT headers path")
set(TensorRT_LIBRARIES "/usr/lib/x86_64-linux-gnu" CACHE STRING "TensorRT libs path")

# DeepStream
set(DEEPSTREAM "/opt/nvidia/deepstream/deepstream" CACHE STRING "DeepStream root path")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${DS_INCLUDE_DIRS})

add_library(
        ${PROJECT_NAME}
        SHARED
        custom_mmyolo_bbox_parser/nvdsparsebbox_mmyolo.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin)
