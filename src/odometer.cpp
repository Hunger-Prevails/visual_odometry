# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <ranges>
# include <indicators/progress_bar.hpp>
# include "odometer.hpp"
# include "image_loader.hpp"
# include "feature_matcher.hpp"
# include "feature_extractor.hpp"


Odometer::Odometer(Eigen::Matrix3f intrinsics, std::shared_ptr<ImageLoader> loader, fs::path write_path, int temporal_baseline, int count_features):
    is_initialized(false), intrinsics(intrinsics), loader(loader), write_path(write_path), temporal_baseline(temporal_baseline)
{
    if (loader->size() <= temporal_baseline) {
        throw std::invalid_argument("temporal_baseline must be at least 1");
    }
    extractor = std::make_unique<Extractor>(count_features);

    matcher = std::make_unique<Matcher>();

    std::filesystem::create_directories(write_path);
}

Odometer::~Odometer() = default;

void Odometer::initialize() {
    is_initialized = true;

    auto image_a = loader->operator[](0);
    auto image_b = loader->operator[](temporal_baseline);

    rotations.emplace(0, Eigen::Quaternionf::Identity());
    translations.emplace(0, Eigen::Vector3f::Zero());

    std::vector<cv::KeyPoint> keypoints_a, keypoints_b;
    cv::Mat descriptors_a, descriptors_b;

    extractor->extract(image_a, keypoints_a, descriptors_a);
    extractor->extract(image_b, keypoints_b, descriptors_b);

    std::vector<cv::DMatch> matches;

    matcher->match(descriptors_a, descriptors_b, matches);

    std::cout << "Initial matches found: " << matches.size() << std::endl;

    matcher->paint_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches,
        write_path / "initial_matches.png"
    );
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
    indicators::ProgressBar bar_a{
        indicators::option::MaxProgress{temporal_baseline - 1},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
        indicators::option::PrefixText{"process baseline frames:"}
    };
    for (size_t i = 1; i < temporal_baseline; ++i) {
        bar_a.tick();
        processFrame(i);
    }

    indicators::ProgressBar bar_b{
        indicators::option::MaxProgress{loader->size() - temporal_baseline},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
        indicators::option::PrefixText{"process subsequent frames:"}
    };
    for (size_t i = temporal_baseline; i < loader->size(); ++i) {
        bar_b.tick();
        processFrame(i);
    }

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
