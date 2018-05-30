cmake_minimum_required(VERSION 2.8.3)
project(koralROS)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  std_msgs
  image_transport
)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -std=c++11 -march=native -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize -mavx2 -mfma -fomit-frame-pointer")
SET(CUDA_NVCC_FLAGS "-arch=sm_61 --use_fast_math -O3" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 
include_directories(include)
CUDA_ADD_LIBRARY(koral ${LIB_TYPE} src/CUDALERP.cu src/CLATCH.cu src/CUDAK2NN.cu)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES koralROS
  CATKIN_DEPENDS cv_bridge roscpp std_msgs image_transport
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(koralNode src/koral_node.cpp)

add_dependencies(koralNode koral)

target_link_libraries(koralNode
   ${catkin_LIBRARIES}
  koral
  ${OpenCV_LIBS}
 )
