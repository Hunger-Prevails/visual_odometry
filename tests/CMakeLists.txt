find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(PCL CONFIG REQUIRED COMPONENTS COMMON IO FILTERS)

add_executable(test test.cpp)
if(MSVC)
    target_compile_options(test PRIVATE /EHsc)
endif()
target_link_libraries(
    test PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        opencv_core
        opencv_features2d
        opencv_imgproc
        igl::igl_core
        pcl_COMMON
        pcl_IO
        pcl_FILTERS
)
