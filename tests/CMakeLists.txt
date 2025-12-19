find_package(PCL REQUIRED COMPONENTS COMMON IO FILTERS)
find_package(Eigen3 REQUIRED)
find_package(libigl REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(test test.cpp)
if(MSVC)
    target_compile_options(test PRIVATE /EHsc)
endif()
target_link_libraries(
    test PRIVATE
        pcl_COMMON
        pcl_IO
        pcl_FILTERS
        Eigen3::Eigen
        igl::igl_core
        opencv_core
        opencv_features2d
        opencv_imgproc
)
