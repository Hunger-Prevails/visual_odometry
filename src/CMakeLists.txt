find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(indicators CONFIG REQUIRED)

add_library(odometer odometer.cpp image_loader.cpp)
target_include_directories(odometer PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(odometer PRIVATE /EHsc)
endif()
target_link_libraries(
    odometer PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        opencv_core
        opencv_features2d
        opencv_imgproc
        opencv_imgcodecs
        indicators::indicators
)
