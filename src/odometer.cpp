# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/core/eigen.hpp>
# include <ranges>
# include <numbers>
# include <ceres/ceres.h>
# include <indicators/progress_bar.hpp>
# include "odometer.hpp"
# include "image_loader.hpp"
# include "feature_matcher.hpp"
# include "feature_extractor.hpp"
# include "cost_functions.hpp"
# include "viz_tools.hpp"
# include "utils.hpp"

const int Odometer::perspective_count(10);
const int Odometer::perspective_iterations(100);
const float Odometer::perspective_error(2.0);
const float Odometer::perspective_error_rescue(3.0);
const float Odometer::perspective_confidence(0.99);

const Eigen::Quaterniond Odometer::rotation_initial = Eigen::Quaterniond::Identity();
const Eigen::Vector3d Odometer::translation_initial = Eigen::Vector3d::Zero();

bool Odometer::skip_keyframe(bool allow_keyframe, int track_count, float track_ratio, float median_pixel_motion) const {
    if (!allow_keyframe)
        return true;

    if (this->track_count < track_count && this->track_ratio < track_ratio) {
        std::cout << "=> => can still track an healthy amount of features" << std::endl;
        return true;
    }

    if (median_pixel_motion <= this->median_pixel_motion) {
        std::cout << "=> => median pixel motion is too low" << std::endl;
        return true;
    }

    return false;
}

Odometer::Odometer(
    Eigen::Matrix3d intrinsics,
    std::unique_ptr<ImageLoader> loader,
    fs::path write_path,
    int count_features,
    int count_keyframes,
    int temporal_baseline,
    float essential_confidence,
    float essential_error,
    float essential_error_initial,
    float tolerance_function,
    float tolerance_gradient,
    float tolerance_parameter,
    int solver_iterations,
    float test_ratio,
    int track_count,
    float track_ratio,
    float median_pixel_motion,
    float parallax_angle
):
    is_initialized(false),
    intrinsics(intrinsics),
    loader(std::move(loader)),
    write_path(write_path),
    count_keyframes(count_keyframes),
    temporal_baseline(temporal_baseline),
    essential_confidence(essential_confidence),
    essential_error(essential_error),
    essential_error_initial(essential_error_initial),
    tolerance_function(tolerance_function),
    tolerance_gradient(tolerance_gradient),
    tolerance_parameter(tolerance_parameter),
    solver_iterations(solver_iterations),
    track_count(track_count),
    track_ratio(track_ratio),
    median_pixel_motion(median_pixel_motion),
    parallax_angle(parallax_angle)
{
    if (this->loader->size() <= temporal_baseline) {
        throw std::invalid_argument("There has to be at least as many frames as the temporal_baseline");
    }
    extractor = std::make_unique<Extractor>(count_features);

    matcher = std::make_unique<Matcher>(test_ratio);

    std::filesystem::create_directories(write_path);
}

Odometer::~Odometer() = default;

void Odometer::show_keyframe(
    const std::shared_ptr<Keyframe> keyframe,
    const fs::path& write_path
) const {
    auto image = loader->operator[](keyframe->frame);

    show_projections(
        image,
        keyframe->feature_to_landmark,
        keyframe->keypoints,
        this->landmarks,
        intrinsics,
        rotations.at(keyframe->frame),
        translations.at(keyframe->frame),
        write_path
    );
}

std::vector<bool> Odometer::landmarks_to_freeze(
    const std::shared_ptr<Keyframe> keyframe
) const {
    std::vector<bool> to_freeze(landmarks.size(), false);

    std::ranges::for_each(
        keyframe->feature_to_landmark | std::views::values,
        [&] (int landmark) {
            to_freeze[landmark] = true;
        }
    );
    return to_freeze;
}

std::unordered_map<int, int> Odometer::create_map(const std::vector<cv::DMatch>& matches, std::shared_ptr<Keyframe> keyframe) const {
    std::unordered_map<int, int> feature_to_landmark;

    for (size_t i = 0; i < matches.size(); ++i) {
        auto it = keyframe->feature_to_landmark.find(matches[i].queryIdx);

        if (it == keyframe->feature_to_landmark.end()) {
            throw std::runtime_error("match does not correspond to any existant landmark");
        }
        feature_to_landmark.emplace(matches[i].trainIdx, it->second);
    }
    return feature_to_landmark;
}

std::vector<cv::DMatch> Odometer::pick_matches_to_track(
    const std::shared_ptr<Keyframe> keyframe,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::DMatch> matches_to_track;

    for (auto& match: matches) {
        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator != keyframe->feature_to_landmark.end()) matches_to_track.push_back(match);
    }
    return matches_to_track;
}

std::vector<cv::DMatch> Odometer::pick_matches_to_chart(
    const std::shared_ptr<Keyframe> keyframe,
    const std::shared_ptr<Keyframe> newframe,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::DMatch> matches_to_chart;

    for (auto& match: matches) {
        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator != keyframe->feature_to_landmark.end()) continue;

        iterator = newframe->feature_to_landmark.find(match.trainIdx);

        if (iterator != newframe->feature_to_landmark.end()) continue;

        matches_to_chart.push_back(match);
    }
    return matches_to_chart;
}

void Odometer::initialize() {
    is_initialized = true;

    auto image_a = loader->operator[](0);
    auto image_b = loader->operator[](temporal_baseline);

    auto [keypoints_a, descriptors_a] = extractor->extract(image_a);
    auto [keypoints_b, descriptors_b] = extractor->extract(image_b);

    auto matches = matcher->match_knn(descriptors_a, descriptors_b);

    std::cout << "=> => finds [" << matches.size() << "] initial matches" << std::endl;

    paint_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches,
        write_path / "matches_initial.png"
    );

    auto [rotation_a, rotation_b, translation_a, translation_b, matches_inliers] = compute_pose_initial(keypoints_a, keypoints_b, matches);

    paint_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches_inliers,
        write_path / "matches_initial_inliers.png"
    );

    auto [landmarks, matches_viable] = triangulate(
        keypoints_a,
        keypoints_b,
        matches_inliers,
        rotation_a,
        rotation_b,
        translation_a,
        translation_b
    );
    std::cout << "=> => triangulates " << landmarks.size() << " new landmarks" << std::endl;

    auto map_a = create_map_query(matches_viable);
    auto map_b = create_map_train(matches_viable);

    this->landmarks = std::move(landmarks);

    auto frame_a = std::make_shared<Keyframe>(0, true, keypoints_a, descriptors_a, map_a);
    auto frame_b = std::make_shared<Keyframe>(temporal_baseline, true, keypoints_b, descriptors_b, map_b);

    rotations.emplace(0, rotation_a);
    rotations.emplace(temporal_baseline, rotation_b);
    translations.emplace(0, translation_a);
    translations.emplace(temporal_baseline, translation_b);

    show_keyframe(
        frame_a,
        write_path / "projections_initial_a.png"
    );
    show_keyframe(
        frame_b,
        write_path / "projections_initial_b.png"
    );

    bundle_adjustment_initial(frame_a, frame_b);

    show_keyframe(
        frame_a,
        write_path / "projections_initial_bundle_adjustment_a.png"
    );
    show_keyframe(
        frame_b,
        write_path / "projections_initial_bundle_adjustment_b.png"
    );

    this->keyframes.push_back(frame_a);
    this->keyframes.push_back(frame_b);

    std::cout << "=> completes initialization with " << this->landmarks.size() << " landmarks" << std::endl;
}

void Odometer::perhaps_add_to_map(const std::shared_ptr<Keyframe> keyframe, const std::shared_ptr<Keyframe> newframe) {
    std::cout << "=> => attempts to chart against frame [" << keyframe->frame << "]" << std::endl;

    auto matches = matcher->match_knn(keyframe->descriptors, newframe->descriptors);

    std::cout << "=> => => finds [" << matches.size() << "] initial matches" << std::endl;

    auto matches_to_chart = pick_matches_to_chart(keyframe, newframe, matches);

    std::cout << "=> => => picks " << matches_to_chart.size() << " matches to chart" << std::endl;

    auto matches_to_chart_inliers = epipolar_check(
        keyframe->keypoints,
        newframe->keypoints,
        matches_to_chart,
        rotations.at(keyframe->frame),
        rotations.at(newframe->frame),
        translations.at(keyframe->frame),
        translations.at(newframe->frame)
    );

    auto matches_to_chart_parallax = parallax_angle_check(
        keyframe->keypoints,
        newframe->keypoints,
        matches_to_chart_inliers,
        rotations.at(keyframe->frame),
        rotations.at(newframe->frame)
    );

    if (matches_to_chart_parallax.empty()) {
        std::cout << "=> => => no feature match survives parallax angle check" << std::endl;
        return;
    }

    auto [landmarks, matches_to_chart_viable] = triangulate(
        keyframe->keypoints,
        newframe->keypoints,
        matches_to_chart_parallax,
        rotations.at(keyframe->frame),
        rotations.at(newframe->frame),
        translations.at(keyframe->frame),
        translations.at(newframe->frame)
    );
    if (landmarks.empty()) {
        std::cout << "=> => => no new landmark survives cheirality check" << std::endl;
        return;
    }
    newframe->with_triangulation = true;

    std::cout << "=> => => triangulates " << landmarks.size() << " new landmarks" << std::endl;

    auto map_to_chart_keyframe = create_map_query(matches_to_chart_viable, this->landmarks.size());
    auto map_to_chart_newframe = create_map_train(matches_to_chart_viable, this->landmarks.size());

    this->landmarks.insert(this->landmarks.end(), landmarks.begin(), landmarks.end());

    keyframe->feature_to_landmark.insert(map_to_chart_keyframe.begin(), map_to_chart_keyframe.end());
    newframe->feature_to_landmark.insert(map_to_chart_newframe.begin(), map_to_chart_newframe.end());
}

void Odometer::process_frame(int frame, bool allow_keyframe) {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    std::cout << std::endl << std::endl;
    std::cout << "=> to process frame [" << frame << "] <" << loader->get_filename(frame) << ">" << std::endl;

    auto image = loader->operator[](frame);
    auto keyframe = keyframes.back();

    auto [keypoints, descriptors] = extractor->extract(image);

    auto matches = matcher->match_knn(keyframe->descriptors, descriptors);

    std::cout << "=> => finds [" << matches.size() << "] initial matches" << std::endl;

    auto matches_to_track = pick_matches_to_track(keyframe, matches);

    std::cout << "=> => picks " << matches_to_track.size() << " matches to track" << std::endl;

    if (matches_to_track.size() < Odometer::perspective_count) {
        std::cout << "=> not enough matches to track on frame [" << frame << "]" << std::endl;

        throw std::runtime_error("Not enough matches to track => will terminate pipeline run");
    }

    auto [rotation, translation, matches_to_track_inliers, matches_to_track_outliers] = compute_pose(
        keyframe,
        keypoints,
        matches_to_track,
        rotations.at(frame - 1),
        translations.at(frame - 1)
    );

    auto matches_to_track_rescue = rescue_matches(keyframe, keypoints, matches_to_track_outliers, rotation, translation);

    matches_to_track_inliers.insert(matches_to_track_inliers.end(), matches_to_track_rescue.begin(), matches_to_track_rescue.end());

    rotations.emplace(frame, rotation);
    translations.emplace(frame, translation);

    std::cout << "=> => was able to track [" << matches_to_track_inliers.size() << " | ";
    std::cout << keyframe->feature_to_landmark.size() << "] landmarks" << std::endl;

    auto track_count = matches_to_track_inliers.size();
    auto track_ratio = float(track_count) / float(keyframe->feature_to_landmark.size());

    auto median_pixel_motion = compute_median_pixel_motion(keyframe->keypoints, keypoints, matches_to_track_inliers);

    if (skip_keyframe(allow_keyframe, track_count, track_ratio, median_pixel_motion)) {
        std::cout << "=> to skip keyframe creation for frame " << frame << std::endl;
        return;
    }
    auto map_to_track = create_map(matches_to_track_inliers, keyframe);

    auto newframe = std::make_shared<Keyframe>(frame, false, keypoints, descriptors, map_to_track);

    std::cout << std::endl;
    std::cout << "=> registers new keyframe [" << frame << "] with " << newframe->feature_to_landmark.size() << " landmarks" << std::endl;

    show_keyframe(
        newframe,
        write_path / ("projections_frame_" + std::to_string(frame) + ".png")
    );

    auto keyframes_with_triangulation = keyframes | std::views::filter(
        [] (const std::shared_ptr<Keyframe>& keyframe) {
            return keyframe->with_triangulation;
        }
    );

    auto keyframes_without_triangulation = keyframes | std::views::filter(
        [] (const std::shared_ptr<Keyframe>& keyframe) {
            return !keyframe->with_triangulation;
        }
    );

    for (auto keyframe: keyframes_with_triangulation) perhaps_add_to_map(keyframe, newframe);

    for (auto keyframe: keyframes_without_triangulation) perhaps_add_to_map(keyframe, newframe);

    keyframes.push_back(newframe);

    if (!newframe->with_triangulation) {
        std::cout << "=> no new landmarks => to skip local bundle adjustment" << std::endl;
        return;
    }
    bundle_adjustment();

    show_keyframe(
        newframe,
        write_path / ("projections_frame_" + std::to_string(newframe->frame) + "_bundle_adjustment.png")
    );
    while (count_keyframes < keyframes.size()) keyframes.pop_front();
}

void Odometer::process_frames() {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }

    indicators::ProgressBar bar_a{
        indicators::option::MaxProgress{temporal_baseline - 1},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
    };
    for (size_t i = 1; i < temporal_baseline; ++i) {
        bar_a.tick();
        process_frame(i, false);
    }

    indicators::ProgressBar bar_b{
        indicators::option::MaxProgress{loader->size() - temporal_baseline},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
    };
    for (size_t i = temporal_baseline + 1; i < loader->size(); ++i) {
        bar_b.tick();
        process_frame(i);
    }

    std::cout << "=> completes visual odometry" << std::endl;
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

    for (auto& match: matches) {
        points_a.push_back(keypoints_a[match.queryIdx].pt);
        points_b.push_back(keypoints_b[match.trainIdx].pt);
    }

    return {points_a, points_b};
}

std::tuple<std::vector<cv::Point2f>, Eigen::MatrixX3d> Odometer::keypoints_to_landmarks(
    const std::shared_ptr<Keyframe> keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::Point2f> points;

    points.reserve(matches.size());

    Eigen::MatrixX3d landmarks(matches.size(), 3);

    for (size_t i = 0; i < matches.size(); ++i) {
        auto& match = matches[i];

        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator == keyframe->feature_to_landmark.end()) {
            throw std::runtime_error("All matches must correspond to existant landmarks.");
        }
        points.push_back(keypoints[match.trainIdx].pt);

        landmarks.row(i) = this->landmarks[iterator->second];
    }

    return {points, landmarks};
}

std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Vector3d, Eigen::Vector3d, std::vector<cv::DMatch>> Odometer::compute_pose_initial(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches
) const {
    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    cv::Mat mask;
    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);
    cv::Mat essentials = cv::findEssentialMat(
        points_a,
        points_b,
        intrinsics,
        cv::RANSAC,
        Odometer::essential_confidence,
        Odometer::essential_error_initial,
        mask
    );

    if (essentials.empty()) {
        throw std::invalid_argument("Cannot compute Essential Matrix from given points");
    }

    cv::Mat rotation;
    cv::Mat translation;

    int inliers = cv::recoverPose(essentials, points_a, points_b, intrinsics, rotation, translation, mask);

    std::cout << "=> => finds " << inliers << " inlier keypoint matches after essential matrix recovery" << std::endl;

    auto matches_inliers = funnel_matches(matches, mask);

    auto [rotation_eigen, translation_eigen] = to_eigen(rotation, translation);

    auto rotation_a = Odometer::rotation_initial;
    auto rotation_b = rotation_eigen * rotation_a;

    auto translation_a = Odometer::translation_initial;
    auto translation_b = translation_eigen + rotation_eigen * translation_a;

    return {
        rotation_a, rotation_b, translation_a, translation_b, matches_inliers
    };
}

std::tuple<Eigen::Quaterniond, Eigen::Vector3d, std::vector<cv::DMatch>, std::vector<cv::DMatch>> Odometer::compute_pose(
    const std::shared_ptr<Keyframe> keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation
) const {
    auto [points, landmarks] = keypoints_to_landmarks(keyframe, keypoints, matches);

    auto [rotation_mat, translation_mat] = from_eigen(rotation, translation);

    auto landmarks_mat = from_eigen(landmarks);

    cv::Mat inliers;
    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);

    cv::Mat rotation_vector;
    cv::Rodrigues(rotation_mat, rotation_vector);

    auto success = cv::solvePnPRansac(
        landmarks_mat,
        points,
        intrinsics,
        cv::Mat(),
        rotation_vector,
        translation_mat,
        true,
        Odometer::perspective_iterations,
        Odometer::perspective_error,
        Odometer::perspective_confidence,
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );
    if (!success) {
        throw std::runtime_error("Cannot compute camera pose from given points and landmarks");
    }
    cv::Rodrigues(rotation_vector, rotation_mat);

    std::cout << "=> => finds " << inliers.rows << " inlier keypoint matches after perspective-n-point" << std::endl;

    auto [rotation_eigen, translation_eigen] = to_eigen(rotation_mat, translation_mat);

    auto [matches_inliers, matches_outliers] = select_matches(matches, inliers);

    return {rotation_eigen, translation_eigen, matches_inliers, matches_outliers};
}

std::vector<cv::DMatch> Odometer::epipolar_check(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation_a,
    const Eigen::Quaterniond& rotation_b,
    const Eigen::Vector3d& translation_a,
    const Eigen::Vector3d& translation_b
) const {
    auto essentials = to_essentials(rotation_a, rotation_b, translation_a, translation_b);

    auto fundamentals = intrinsics.inverse().transpose() * essentials * intrinsics.inverse();

    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    auto product = epipolar_products(fundamentals, to_homogeneous(points_a), to_homogeneous(points_b));

    auto inliers = (product.array() < Odometer::essential_error).matrix();

    std::vector<cv::DMatch> matches_inliers;

    for (size_t i = 0; i < matches.size(); ++i) {
        if (!inliers(i)) continue;

        matches_inliers.push_back(matches[i]);
    }
    std::cout << "=> => => finds " << matches_inliers.size() << " inlier keypoint matches after epipolar check" << std::endl;

    return matches_inliers;
}

std::vector<cv::DMatch> Odometer::rescue_matches(
    const std::shared_ptr<Keyframe> keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation
) const {
    std::vector<cv::DMatch> matches_to_rescue;

    auto [points, landmarks] = keypoints_to_landmarks(keyframe, keypoints, matches);

    auto points_eigen = to_eigen(points);

    auto landmarks_camera = (rotation.toRotationMatrix() * landmarks.transpose()).colwise() + translation;

    auto projections = hnormalize((intrinsics * landmarks_camera).transpose());

    auto distance = (projections - points_eigen).rowwise().norm();

    auto mask = (distance.array() < Odometer::perspective_error_rescue).matrix();

    for (size_t i = 0; i < matches.size(); ++i) {
        if (!mask(i)) continue;

        matches_to_rescue.push_back(matches[i]);
    }
    std::cout << "=> => to rescue " << matches_to_rescue.size() << " matches after perspective-n-point reprojection check" << std::endl;

    return matches_to_rescue;
}

std::vector<cv::DMatch> Odometer::parallax_angle_check(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation_a,
    const Eigen::Quaterniond& rotation_b
) const {
    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    auto points_a_homogeneous = to_homogeneous(points_a);
    auto points_b_homogeneous = to_homogeneous(points_b);

    auto rays_a_camera = points_a_homogeneous * intrinsics.inverse().transpose();
    auto rays_b_camera = points_b_homogeneous * intrinsics.inverse().transpose();

    Eigen::MatrixX3d rays_a_world = rays_a_camera * rotation_a.toRotationMatrix();
    Eigen::MatrixX3d rays_b_world = rays_b_camera * rotation_b.toRotationMatrix();

    auto dot_products = (rays_a_world.array() * rays_b_world.array()).rowwise().sum();

    auto norms_a = rays_a_world.rowwise().norm();
    auto norms_b = rays_b_world.rowwise().norm();

    Eigen::ArrayXd cosines = dot_products.array() / norms_a.array() / norms_b.array();

    auto angles = cosines.acos() / (std::numbers::pi / 180.0);

    std::vector<cv::DMatch> matches_parallax;

    for (size_t i = 0; i < matches.size(); ++i) if (parallax_angle <= angles(i)) matches_parallax.push_back(matches[i]);

    return matches_parallax;
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<cv::DMatch>> Odometer::triangulate(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation_a,
    const Eigen::Quaterniond& rotation_b,
    const Eigen::Vector3d& translation_a,
    const Eigen::Vector3d& translation_b
) const {
    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);

    auto [rotation_a_mat, translation_a_mat] = from_eigen(rotation_a, translation_a);
    auto [rotation_b_mat, translation_b_mat] = from_eigen(rotation_b, translation_b);

    cv::Mat extrinsics_a;
    cv::Mat extrinsics_b;

    cv::hconcat(rotation_a_mat, translation_a_mat, extrinsics_a);
    cv::hconcat(rotation_b_mat, translation_b_mat, extrinsics_b);

    auto projection_a = intrinsics * extrinsics_a;
    auto projection_b = intrinsics * extrinsics_b;

    cv::Mat points_homogeneous;
    cv::triangulatePoints(projection_a, projection_b, points_a, points_b, points_homogeneous);

    std::vector<Eigen::Vector3d> landmarks;

    landmarks.reserve(points_homogeneous.cols);

    std::vector<cv::DMatch> matches_viable;

    matches_viable.reserve(points_homogeneous.cols);

    for (int i = 0; i < points_homogeneous.cols; i ++) {
        float w = points_homogeneous.at<float>(3, i);

        if (std::abs(w) < 1e-5) continue;

        auto x = points_homogeneous.at<float>(0, i) / w;
        auto y = points_homogeneous.at<float>(1, i) / w;
        auto z = points_homogeneous.at<float>(2, i) / w;

        auto landmark = Eigen::Vector3d(x, y, z);

        if ((rotation_a * landmark + translation_a).z() < 0.0) continue;
        if ((rotation_b * landmark + translation_b).z() < 0.0) continue;

        landmarks.push_back(landmark);
        matches_viable.push_back(matches[i]);
    }
    return {landmarks, matches_viable};
}

void Odometer::bundle_adjustment_initial(std::shared_ptr<Keyframe> frame_a, std::shared_ptr<Keyframe> frame_b) {
    auto problem = ceres::Problem();
    auto loss_function = new ceres::HuberLoss(1.0);

    for (auto [feature, landmark]: frame_a->feature_to_landmark) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionErrorLandmark, 2, 3>(
                new ProjectionErrorLandmark(
                    frame_a->keypoints[feature].pt, intrinsics, rotations.at(frame_a->frame), translations.at(frame_a->frame)
                )
            ),
            loss_function,
            landmarks[landmark].data()
        );
    }
    problem.AddParameterBlock(rotations.at(frame_b->frame).coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(translations.at(frame_b->frame).data(), 3, new ceres::SphereManifold<3>());

    for (auto [feature, landmark]: frame_b->feature_to_landmark) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionError, 2, 3, 4, 3>(
                new ProjectionError(
                    frame_b->keypoints[feature].pt, intrinsics
                )
            ),
            loss_function,
            landmarks[landmark].data(),
            rotations.at(frame_b->frame).coeffs().data(),
            translations.at(frame_b->frame).data()
        );
    }
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = tolerance_function;
    options.gradient_tolerance = tolerance_gradient;
    options.parameter_tolerance = tolerance_parameter;
    options.max_num_iterations = solver_iterations;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void Odometer::bundle_adjustment() {
    std::cout << std::endl;
    std::cout << "=> to perform local bundle adjustment with " << keyframes.size() << " keyframes" << std::endl;

    auto to_freeze = landmarks_to_freeze(keyframes.front());

    auto problem = ceres::Problem();
    auto loss_function = new ceres::HuberLoss(1.0);

    for (auto keyframe: keyframes) {
        auto& rotation = rotations.at(keyframe->frame);
        auto& translation = translations.at(keyframe->frame);

        problem.AddParameterBlock(rotation.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(translation.data(), 3);

        for (auto [feature, landmark]: keyframe->feature_to_landmark) {
            if (to_freeze[landmark]) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ProjectionErrorTransform, 2, 4, 3>(
                        new ProjectionErrorTransform(
                            keyframe->keypoints[feature].pt, intrinsics, landmarks[landmark]
                        )
                    ),
                    loss_function,
                    rotation.coeffs().data(),
                    translation.data()
                );
            } else {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ProjectionError, 2, 3, 4, 3>(
                        new ProjectionError(
                            keyframe->keypoints[feature].pt, intrinsics
                        )
                    ),
                    loss_function,
                    landmarks[landmark].data(),
                    rotation.coeffs().data(),
                    translation.data()
                );
            }
        }
    }
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = tolerance_function;
    options.gradient_tolerance = tolerance_gradient;
    options.parameter_tolerance = tolerance_parameter;
    options.max_num_iterations = solver_iterations;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return;
}
