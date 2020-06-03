cmake_minimum_required(VERSION 3.1) 
project(yolov3-trt)            

file(GLOB_RECURSE sources     	modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h)
add_executable(yolov3-trt samples/sample_detector.cpp)

set(TENSORRT_ROOT "/home/aipc/code-YK/TensorRT-5.1.5.0")
set(OpenCV_DIR "/usr/local/opencv346")

#find_Package(gflags REQUIRED)

#if(gflags_FOUND)
#message("-- gflags found")
#endif()

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#cudaS
find_package(CUDA REQUIRED)

message("-- CUDA version:$(CUDA_VERSION)")

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected")
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_53,code=sm_53
                -gencode arch=compute_60,code=sm_60
                -gencode arch=compute_61,code=sm_61
                -gencode arch=compute_70,code=sm_70
                -gencode arch=compute_75,code=sm_75
	)
find_package(OpenCV REQUIRED)
endif()

message("-- sources:$(sources)")
cuda_add_library(detector SHARED ${sources})
# /usr/include/gflags/include
target_include_directories(detector PRIVATE modules/  ${CUDA_TOOLKIT_ROOT_DIR}/include /home/aipc/code-YK/TensorRT-5.1.5.0/include /usr/local/opencv346/include extra/ ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#target_link_libraries(detector opencv_core opencv_calib3d opencv_dnn opencv_imgproc opencv_highgui)

target_include_directories(yolov3-trt PRIVATE modules/ extra/)

#LINK_LIBRARIES("/home/yaokun/code/TensorRT-5.1.5.0/lib/libnvcaffe_parser.so")
#LINK_LIBRARIES("/home/yaokun/code/TensorRT-5.1.5.0/lib/libnvparsers.so")
#LINK_LIBRARIES("/home/yaokun/code/TensorRT-5.1.5.0/lib/libnvinfer.so")
#LINK_LIBRARIES("/home/yaokun/code/TensorRT-5.1.5.0/lib/libnvinfer_plugin.so")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
      HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} 
      PATH_SUFFIXES include) 
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}") 
find_library(TENSORRT_LIBRARY_INFER nvinfer 
      HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64 lib/x64) 
#find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin 
#      HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} 
#      PATH_SUFFIXES lib lib64 lib/x64) 
find_library(TENSORRT_LIBRARY_CAFFEPARSER nvcaffe_parser 
      HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin 
      HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64 lib/x64)  
 
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_CAFFEPARSER} ${TENSORRT_LIBRARY_PLUGIN})
MESSAGE(STATUS "TENSORRT_LIBRARY are ${TENSORRT_LIBRARY}")
#target_link_libraries(detector nvinfer nvinfer_plugin nvcaffe_parser gflags "stdc++fs")
target_link_libraries(detector  ${TENSORRT_LIBRARY} "stdc++fs")
				  
if(CUDA_VERSION_MAJOR GREATER 9)
	target_link_libraries(yolov3-trt detector opencv_core opencv_calib3d opencv_dnn opencv_video)
endif()





