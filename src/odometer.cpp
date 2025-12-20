# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <ranges>
# include "odometer.hpp"
# include "image_loader.hpp"

Odometer::Odometer(Eigen::Matrix3f intrinsics, std::shared_ptr<ImageLoader> loader, int temporal_baseline):
    is_initialized(false), intrinsics(intrinsics), loader(loader), temporal_baseline(temporal_baseline)
{
    if (loader->size() <= temporal_baseline) {
        throw std::invalid_argument("temporal_baseline must be at least 1");
    }
}

void Odometer::initialize() {
    is_initialized = true;

    auto image_a = loader->operator[](0);
    auto image_b = loader->operator[](temporal_baseline);

    rotations.emplace(0, Eigen::Quaternionf::Identity());
    translations.emplace(0, Eigen::Vector3f::Zero());

    rotations.emplace(temporal_baseline, Eigen::Quaternionf::Identity());
    translations.emplace(temporal_baseline, Eigen::Vector3f::Zero());

    std::cout << "Completes initialization" << std::endl;
}

void Odometer::processFrame(int index) {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    auto image = loader->operator[](index);

    rotations.emplace(index, Eigen::Quaternionf::Identity());
    translations.emplace(index, Eigen::Vector3f::Zero());
}

void Odometer::processFrames() {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    for (size_t i = 1; i < temporal_baseline; ++i) processFrame(i);
    for (size_t i = temporal_baseline; i < loader->size(); ++i) processFrame(i);

    std::cout << "Completes visual odometry" << std::endl;
}

const std::vector<Eigen::Quaternionf> Odometer::getRotations() {
    std::vector<Eigen::Quaternionf> result;
    result.reserve(rotations.size());

    std::ranges::copy(rotations | std::views::values, std::back_inserter(result));

    return result;
}

const std::vector<Eigen::Vector3f> Odometer::getTranslations() {
    std::vector<Eigen::Vector3f> result;
    result.reserve(translations.size());

    std::ranges::copy(translations | std::views::values, std::back_inserter(result));

    return result;
}
