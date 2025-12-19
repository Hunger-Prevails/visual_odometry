find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
if(MSVC)
    target_compile_options(main PRIVATE /EHsc)
endif()
target_link_libraries(main PRIVATE odometer Eigen3::Eigen opencv_core opencv_features2d opencv_imgproc)
