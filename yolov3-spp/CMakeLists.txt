cmake_minimum_required(VERSION 3.0)

project(yolov3-spp VERSION 1.0)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/opt/cuda/include)
link_directories(targets/x86_64-linux/lib)
# tensorrt
include_directories(/media/bkhti4/External/CUDAshit/TensorRT-8.0.0.3/include)
link_directories(/media/bkhti4/External/CUDAshit/TensorRT-8.0.0.3/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov3-spp ${PROJECT_SOURCE_DIR}/yolov3-spp.cpp)
target_link_libraries(yolov3-spp nvinfer)
target_link_libraries(yolov3-spp cudart)
target_link_libraries(yolov3-spp yololayer)
target_link_libraries(yolov3-spp ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

