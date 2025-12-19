cmake_minimum_required(VERSION 3.10)
project(test_pcl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(PCL REQUIRED COMPONENTS COMMON IO FILTERS)
find_package(Eigen3 REQUIRED)
find_package(libigl REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
if(MSVC)
    target_compile_options(main PRIVATE /EHsc)
endif()
target_link_libraries(
    main PRIVATE
        pcl_COMMON
        pcl_IO
        pcl_FILTERS
        Eigen3::Eigen
        igl::igl_core
        opencv_core
        opencv_features2d
        opencv_imgproc
)
