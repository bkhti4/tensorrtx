cmake_minimum_required(VERSION 3.0)

project(yolov3-tiny VERSION 1.0)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/opt/cuda/include)
    link_directories(/opt/cuda/lib64/targets/x86_64-linux/lib)
    # tensorrt
    include_directories(/media/bkhti4/External/CUDAshit/TensorRT-8.0.0.3/include)
    link_directories(/media/bkhti4/External/CUDAshit/TensorRT-8.0.0.3/lib)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov3-tiny ${PROJECT_SOURCE_DIR}/yolov3-tiny.cpp)
target_link_libraries(yolov3-tiny nvinfer)
target_link_libraries(yolov3-tiny cudart)
target_link_libraries(yolov3-tiny yololayer)
target_link_libraries(yolov3-tiny ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

