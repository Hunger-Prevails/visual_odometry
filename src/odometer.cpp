# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/core/eigen.hpp>
# include <ranges>
# include <ceres/ceres.h>
# include <indicators/progress_bar.hpp>
# include "odometer.hpp"
# include "image_loader.hpp"
# include "feature_matcher.hpp"
# include "feature_extractor.hpp"
# include "cost_functions.hpp"
# include "viz_tools.hpp"


Odometer::Odometer(
    Eigen::Matrix3d intrinsics,
    std::shared_ptr<ImageLoader> loader,
    fs::path write_path,
    int temporal_baseline,
    int n_keyframes,
    int count_features,
    float test_ratio,
    float function_tolerance
):
    is_initialized(false),
    intrinsics(intrinsics),
    loader(loader),
    write_path(write_path),
    temporal_baseline(temporal_baseline),
    n_keyframes(n_keyframes),
    function_tolerance(function_tolerance)
{
    if (loader->size() <= temporal_baseline) {
        throw std::invalid_argument("there has to be at least as many frames as the temporal_baseline");
    }
    extractor = std::make_unique<Extractor>(count_features);

    matcher = std::make_unique<Matcher>(test_ratio);

    std::filesystem::create_directories(write_path);
}

Odometer::~Odometer() = default;

std::tuple<std::unordered_map<int, int>, std::unordered_map<int, int>, std::vector<cv::DMatch>> Odometer::create_map(
    const std::vector<cv::DMatch>& matches, const cv::Mat& mask
) const {
    std::unordered_map<int, int> map_a;
    std::unordered_map<int, int> map_b;
    std::vector<cv::DMatch> matches_inliers;

    for (size_t i = 0; i < matches.size(); ++i) {
        if (!mask.at<uchar>(i)) continue;

        auto match = matches[i];

        map_a.emplace(match.queryIdx, static_cast<int>(matches_inliers.size()));
        map_b.emplace(match.trainIdx, static_cast<int>(matches_inliers.size()));

        matches_inliers.push_back(match);
    }
    return {map_a, map_b, matches_inliers};
}

void Odometer::initialize() {
    is_initialized = true;

    auto image_a = loader->operator[](0);
    auto image_b = loader->operator[](temporal_baseline);

    rotations.emplace(0, Eigen::Quaterniond::Identity());
    translations.emplace(0, Eigen::Vector3d::Zero());

    auto [keypoints_a, descriptors_a] = extractor->extract(image_a);
    auto [keypoints_b, descriptors_b] = extractor->extract(image_b);

    auto matches = matcher->match_knn(descriptors_a, descriptors_b);

    paint_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches,
        write_path / "initial_matches.png"
    );

    auto [rotation, translation, mask] = compute_pose_initial(keypoints_a, keypoints_b, matches);

    auto [map_a, map_b, matches_inliers] = create_map(matches, mask);

    paint_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches_inliers,
        write_path / "initial_matches_inliers.png"
    );

    auto landmarks = triangulate(
        keypoints_a,
        keypoints_b,
        matches_inliers,
        rotation,
        translation
    );

    auto rotation_eigen = to_eigen_rotation(rotation);
    auto translation_eigen = to_eigen_translation(translation);

    paint_projections(
        image_b,
        keypoints_b,
        landmarks,
        map_b,
        intrinsics,
        rotation_eigen,
        translation_eigen,
        write_path / "projections_b_triangulate.png"
    );

    bundle_adjustment_initial(
        keypoints_a,
        keypoints_b,
        matches_inliers,
        landmarks,
        rotation_eigen,
        translation_eigen
    );

    paint_projections(
        image_b,
        keypoints_b,
        landmarks,
        map_b,
        intrinsics,
        rotation_eigen,
        translation_eigen,
        write_path / "projections_b_bundle_adjustment.png"
    );

    this->landmarks = std::move(landmarks);
    this->keyframes.push(std::make_shared<Keyframe>(0, keypoints_a, descriptors_a, map_a));
    this->keyframes.push(std::make_shared<Keyframe>(temporal_baseline, keypoints_b, descriptors_b, map_b));

    rotations.emplace(temporal_baseline, rotation_eigen);
    translations.emplace(temporal_baseline, translation_eigen);

    std::cout << "Completes initialization" << std::endl;
}

void Odometer::processFrame(int index) {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    auto image = loader->operator[](index);
    auto keyframe = keyframes.back();

    auto [keypoints, descriptors] = extractor->extract(image);

    auto matches = matcher->match_knn(keyframe->descriptors, descriptors);

    rotations.emplace(index, Eigen::Quaterniond::Identity());
    translations.emplace(index, Eigen::Vector3d::Zero());
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

const std::vector<Eigen::Quaterniond> Odometer::getRotations() const {
    std::vector<Eigen::Quaterniond> result;
    result.reserve(rotations.size());

    std::ranges::copy(rotations | std::views::values, std::back_inserter(result));

    return result;
}

const std::vector<Eigen::Vector3d> Odometer::getTranslations() const {
    std::vector<Eigen::Vector3d> result;
    result.reserve(translations.size());

    std::ranges::copy(translations | std::views::values, std::back_inserter(result));

    return result;
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> Odometer::keypoints_to_keypoints(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::Point2f> points_a, points_b;

    for (const auto match: matches) {
        points_a.push_back(keypoints_a[match.queryIdx].pt);
        points_b.push_back(keypoints_b[match.trainIdx].pt);
    }

    return {points_a, points_b};
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<cv::DMatch>, std::vector<cv::DMatch>> Odometer::keypoints_to_landmarks(
    const std::shared_ptr<Keyframe>& keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point3f> landmarks;

    std::vector<cv::DMatch> matches_to_track;
    std::vector<cv::DMatch> matches_to_chart;

    for (auto match: matches) {
        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator == keyframe->feature_to_landmark.end()) {
            matches_to_chart.push_back(match);
            continue;
        }
        matches_to_track.push_back(match);

        points.push_back(keypoints[match.trainIdx].pt);

        auto landmark = this->landmarks[iterator->second];

        landmarks.emplace_back(
            static_cast<float>(landmark.x()),
            static_cast<float>(landmark.y()),
            static_cast<float>(landmark.z())
        );
    }

    return {points, landmarks, matches_to_track, matches_to_chart};
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> Odometer::compute_pose_initial(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches
) const {
    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    cv::Mat mask;
    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);
    cv::Mat essentials = cv::findEssentialMat(points_a, points_b, intrinsics, cv::RANSAC, 0.999, 1.0, mask);

    if (essentials.empty()) {
        throw std::invalid_argument("Cannot compute Essential Matrix from given points");
    }

    cv::Mat rotation;
    cv::Mat translation;

    int inliers = cv::recoverPose(essentials, points_a, points_b, intrinsics, rotation, translation, mask);

    std::cout << "Found " << inliers << " inlier keypoint matches" << std::endl;

    return {
        rotation, translation, mask
    };
}

std::vector<Eigen::Vector3d> Odometer::triangulate(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& rotation,
    const cv::Mat& translation
) const {
    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);

    cv::Mat projection_a = cv::Mat::zeros(3, 4, intrinsics.type());

    intrinsics.copyTo(projection_a(cv::Rect(0, 0, 3, 3)));

    cv::Mat extrinsics;
    cv::hconcat(rotation, translation, extrinsics);

    auto projection_b = intrinsics * extrinsics;

    cv::Mat points_homogeneous;
    cv::triangulatePoints(projection_a, projection_b, points_a, points_b, points_homogeneous);

    std::vector<Eigen::Vector3d> landmarks;

    landmarks.reserve(points_homogeneous.cols);

    for (int i = 0; i < points_homogeneous.cols; i ++) {
        float w = points_homogeneous.at<float>(3, i);

        if (std::abs(w) < 1e-5) continue;

        auto x = points_homogeneous.at<float>(0, i) / w;
        auto y = points_homogeneous.at<float>(1, i) / w;
        auto z = points_homogeneous.at<float>(2, i) / w;

        if (z > 0) landmarks.push_back({x, y, z});
    }
    if (landmarks.size() != matches.size()) {
        throw std::runtime_error("Triangulation was not possible for some matches.");
    }
    return landmarks;
}

Eigen::Quaterniond Odometer::to_eigen_rotation(const cv::Mat& rotation) const {
    Eigen::Matrix3d rotation_eigen;
    cv::cv2eigen(rotation, rotation_eigen);
    return Eigen::Quaterniond(rotation_eigen);
}

Eigen::Vector3d Odometer::to_eigen_translation(const cv::Mat& translation) const {
    Eigen::Vector3d translation_eigen;
    cv::cv2eigen(translation, translation_eigen);
    return translation_eigen;
}

void Odometer::bundle_adjustment_initial(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    std::vector<Eigen::Vector3d>& landmarks,
    Eigen::Quaterniond& rotation,
    Eigen::Vector3d& translation
) const {
    if (landmarks.size() != matches.size()) {
        throw std::invalid_argument("Number of landmarks must be equal to number of matches");
    }

    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    auto problem = ceres::Problem();
    auto loss_function = new ceres::HuberLoss(1.0);

    problem.AddParameterBlock(rotation.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(translation.data(), 3, new ceres::SphereManifold<3>());

    for (size_t i = 0; i < matches.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionError, 2, 3>(new ProjectionError(points_a[i], intrinsics)),
            loss_function,
            landmarks[i].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionErrorFull, 2, 3, 4, 3>(new ProjectionErrorFull(points_b[i], intrinsics)),
            loss_function,
            landmarks[i].data(),
            rotation.coeffs().data(),
            translation.data()
        );
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = function_tolerance;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}
