find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(odometer odometer.cpp)
target_include_directories(odometer PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(odometer PRIVATE /EHsc)
endif()
target_link_libraries(
    odometer PRIVATE
        Eigen3::Eigen
        opencv_core
        opencv_features2d
        opencv_imgproc
)
