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
# include "utils.hpp"

const int Odometer::perspective_iterations(100);
const float Odometer::perspective_error(2.0);
const float Odometer::perspective_confidence(0.99);

const Eigen::Quaterniond Odometer::rotation_initial = Eigen::Quaterniond::Identity();
const Eigen::Vector3d Odometer::translation_initial = Eigen::Vector3d::Zero();

Odometer::Odometer(
    Eigen::Matrix3d intrinsics,
    std::shared_ptr<ImageLoader> loader,
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
    float test_ratio,
    float track_ratio
):
    is_initialized(false),
    intrinsics(intrinsics),
    loader(loader),
    write_path(write_path),
    count_keyframes(count_keyframes),
    temporal_baseline(temporal_baseline),
    essential_confidence(essential_confidence),
    essential_error(essential_error),
    essential_error_initial(essential_error_initial),
    tolerance_function(tolerance_function),
    tolerance_gradient(tolerance_gradient),
    tolerance_parameter(tolerance_parameter),
    track_ratio(track_ratio)
{
    if (loader->size() <= temporal_baseline) {
        throw std::invalid_argument("There has to be at least as many frames as the temporal_baseline");
    }
    extractor = std::make_unique<Extractor>(count_features);

    matcher = std::make_unique<Matcher>(test_ratio);

    std::filesystem::create_directories(write_path);
}

Odometer::~Odometer() = default;

std::vector<bool> Odometer::landmarks_to_freeze(
    const std::shared_ptr<Keyframe>& keyframe
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

std::unordered_map<int, int> Odometer::create_map(const std::vector<cv::DMatch>& matches, std::shared_ptr<Keyframe>& keyframe) const {
    std::unordered_map<int, int> feature_to_landmark;

    for (size_t i = 0; i < matches.size(); ++i) {
        auto it = keyframe->feature_to_landmark.find(matches[i].queryIdx);

        if (it == keyframe->feature_to_landmark.end()) {
            throw std::runtime_error("match does not correspond to any existant landmark");
        }
        feature_to_landmark.emplace(matches[i].trainIdx, it->second);
    }
    std::cout << "Manages to track " << feature_to_landmark.size() << " existant landmarks" << std::endl;

    return feature_to_landmark;
}

std::pair<std::vector<cv::DMatch>, std::vector<cv::DMatch>> Odometer::track_or_chart(
    const std::shared_ptr<Keyframe>& keyframe,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::DMatch> matches_to_track;
    std::vector<cv::DMatch> matches_to_chart;

    for (auto& match: matches) {
        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator == keyframe->feature_to_landmark.end()) {
            matches_to_chart.push_back(match);
            continue;
        }
        matches_to_track.push_back(match);
    }
    std::cout << "to track " << matches_to_track.size() << " matches" << std::endl;
    std::cout << "to chart " << matches_to_chart.size() << " matches" << std::endl;

    return {matches_to_track, matches_to_chart};
}

void Odometer::initialize() {
    is_initialized = true;

    auto image_a = loader->operator[](0);
    auto image_b = loader->operator[](temporal_baseline);

    auto [keypoints_a, descriptors_a] = extractor->extract(image_a);
    auto [keypoints_b, descriptors_b] = extractor->extract(image_b);

    auto matches = matcher->match_knn(descriptors_a, descriptors_b);

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
    auto map_a = create_map_query(matches_viable);
    auto map_b = create_map_train(matches_viable);

    paint_projections(
        image_b,
        keypoints_b,
        landmarks,
        map_b,
        intrinsics,
        rotation_b,
        translation_b,
        write_path / "projections_initial.png"
    );

    bundle_adjustment_initial(
        keypoints_a,
        keypoints_b,
        matches_viable,
        landmarks,
        rotation_a,
        rotation_b,
        translation_a,
        translation_b
    );

    paint_projections(
        image_b,
        keypoints_b,
        landmarks,
        map_b,
        intrinsics,
        rotation_b,
        translation_b,
        write_path / "projections_initial_bundle_adjustment.png"
    );

    this->landmarks = std::move(landmarks);
    this->keyframes.push_back(std::make_shared<Keyframe>(0, keypoints_a, descriptors_a, map_a));
    this->keyframes.push_back(std::make_shared<Keyframe>(temporal_baseline, keypoints_b, descriptors_b, map_b));

    rotations.emplace(0, rotation_a);
    rotations.emplace(temporal_baseline, rotation_b);
    translations.emplace(0, translation_a);
    translations.emplace(temporal_baseline, translation_b);

    std::cout << "Completes initialization with " << this->landmarks.size() << " landmarks" << std::endl;
}

void Odometer::process_frame(int frame, bool allow_keyframe) {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    auto image = loader->operator[](frame);
    auto& keyframe = keyframes.back();

    auto [keypoints, descriptors] = extractor->extract(image);

    auto matches = matcher->match_knn(keyframe->descriptors, descriptors);

    auto [matches_to_track, matches_to_chart] = track_or_chart(keyframe, matches);

    auto [matches_to_track_inliers, rotation, translation] = compute_pose(
        keyframe,
        keypoints,
        matches_to_track,
        rotations.at(frame - 1),
        translations.at(frame - 1)
    );

    auto map_to_track = create_map(matches_to_track_inliers, keyframe);

    rotations.emplace(frame, rotation);
    translations.emplace(frame, translation);

    if (!allow_keyframe || track_ratio <= float(map_to_track.size()) / float(keyframe->feature_to_landmark.size())) {
        std::cout << "Skips keyframe creation for frame " << frame << std::endl;
        return;
    }
    std::cout << "To create keyframe for frame " << frame << std::endl;

    auto newframe = std::make_shared<Keyframe>(frame, keypoints, descriptors, map_to_track);

    auto matches_to_chart_inliers = epipolar_check(
        keyframe->keypoints,
        keypoints,
        matches_to_chart,
        rotations.at(keyframe->frame),
        rotation,
        translations.at(keyframe->frame),
        translation
    );

    auto [landmarks, matches_to_chart_viable] = triangulate(
        keyframe->keypoints,
        keypoints,
        matches_to_chart_inliers,
        rotations.at(keyframe->frame),
        rotation,
        translations.at(keyframe->frame),
        translation
    );
    auto map_to_chart_keyframe = create_map_query(matches_to_chart_viable, this->landmarks.size());
    auto map_to_chart_newframe = create_map_train(matches_to_chart_viable, this->landmarks.size());

    keyframe->feature_to_landmark.insert(map_to_chart_keyframe.begin(), map_to_chart_keyframe.end());
    newframe->feature_to_landmark.insert(map_to_chart_newframe.begin(), map_to_chart_newframe.end());

    this->landmarks.insert(this->landmarks.end(), landmarks.begin(), landmarks.end());
    this->keyframes.push_back(newframe);

    paint_projections(
        loader->operator[](keyframe->frame),
        keyframe->keypoints,
        this->landmarks,
        keyframe->feature_to_landmark,
        intrinsics,
        rotations.at(keyframe->frame),
        translations.at(keyframe->frame),
        write_path / ("projections_frame_" + std::to_string(frame) + "_keyframe_" + std::to_string(keyframe->frame) + ".png")
    );
    paint_projections(
        image,
        keypoints,
        this->landmarks,
        newframe->feature_to_landmark,
        intrinsics,
        rotation,
        translation,
        write_path / ("projections_frame_" + std::to_string(frame) + ".png")
    );

    auto to_freeze = landmarks_to_freeze(keyframes.front());

    if (count_keyframes < keyframes.size()) keyframes.pop_front();

    bundle_adjustment(to_freeze);

    paint_projections(
        loader->operator[](keyframe->frame),
        keyframe->keypoints,
        this->landmarks,
        keyframe->feature_to_landmark,
        intrinsics,
        rotations.at(keyframe->frame),
        translations.at(keyframe->frame),
        write_path / ("projections_frame_" + std::to_string(frame) + "_keyframe_" + std::to_string(keyframe->frame) + "_bundle_adjustment.png")
    );
    paint_projections(
        image,
        keypoints,
        this->landmarks,
        newframe->feature_to_landmark,
        intrinsics,
        rotation,
        translation,
        write_path / ("projections_frame_" + std::to_string(frame) + "_bundle_adjustment.png")
    );
    exit(0);
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
        indicators::option::PrefixText{"process baseline frames:"}
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
        indicators::option::PrefixText{"process subsequent frames:"}
    };
    for (size_t i = temporal_baseline + 1; i < loader->size(); ++i) {
        bar_b.tick();
        process_frame(i);
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

    for (auto& match: matches) {
        points_a.push_back(keypoints_a[match.queryIdx].pt);
        points_b.push_back(keypoints_b[match.trainIdx].pt);
    }

    return {points_a, points_b};
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>> Odometer::keypoints_to_landmarks(
    const std::shared_ptr<Keyframe>& keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches
) const {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point3f> landmarks;

    for (auto& match: matches) {
        auto iterator = keyframe->feature_to_landmark.find(match.queryIdx);

        if (iterator == keyframe->feature_to_landmark.end()) {
            throw std::runtime_error("All matches must correspond to existant landmarks.");
        }
        points.push_back(keypoints[match.trainIdx].pt);

        auto landmark = this->landmarks[iterator->second];

        landmarks.emplace_back(
            static_cast<float>(landmark.x()),
            static_cast<float>(landmark.y()),
            static_cast<float>(landmark.z())
        );
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

    std::cout << "Found " << inliers << " inlier keypoint matches after essential matrix recovery" << std::endl;

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

std::tuple<std::vector<cv::DMatch>, Eigen::Quaterniond, Eigen::Vector3d> Odometer::compute_pose(
    const std::shared_ptr<Keyframe>& keyframe,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation
) const {
    auto [points, landmarks] = keypoints_to_landmarks(keyframe, keypoints, matches);

    auto [rotation_mat, translation_mat] = from_eigen(rotation, translation);

    cv::Mat inliers;
    cv::Mat intrinsics;
    cv::eigen2cv(this->intrinsics, intrinsics);

    cv::Mat rotation_vector;
    cv::Rodrigues(rotation_mat, rotation_vector);

    auto success = cv::solvePnPRansac(
        landmarks,
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

    std::cout << "Found " << inliers.rows << " inlier keypoint matches after perspective-n-point" << std::endl;

    auto [rotation_eigen, translation_eigen] = to_eigen(rotation_mat, translation_mat);

    return {select_matches(matches, inliers), rotation_eigen, translation_eigen};
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
    std::cout << "Found " << matches_inliers.size() << " inlier keypoint matches after epipolar check" << std::endl;

    return matches_inliers;
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
    std::cout << "Manages to triangulate " << landmarks.size() << " landmarks from viable matches" << std::endl;

    return {landmarks, matches_viable};
}

void Odometer::bundle_adjustment_initial(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    std::vector<Eigen::Vector3d>& landmarks,
    Eigen::Quaterniond& rotation_a,
    Eigen::Quaterniond& rotation_b,
    Eigen::Vector3d& translation_a,
    Eigen::Vector3d& translation_b
) const {
    if (landmarks.size() != matches.size()) {
        throw std::invalid_argument("Number of landmarks must be equal to number of matches");
    }

    auto [points_a, points_b] = keypoints_to_keypoints(keypoints_a, keypoints_b, matches);

    auto problem = ceres::Problem();
    auto loss_function = new ceres::HuberLoss(1.0);

    problem.AddParameterBlock(rotation_b.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(translation_b.data(), 3, new ceres::SphereManifold<3>());

    for (size_t i = 0; i < matches.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionErrorLandmark, 2, 3>(
                new ProjectionErrorLandmark(
                    points_a[i], intrinsics, rotation_a, translation_a
                )
            ),
            loss_function,
            landmarks[i].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ProjectionError, 2, 3, 4, 3>(
                new ProjectionError(
                    points_b[i], intrinsics
                )
            ),
            loss_function,
            landmarks[i].data(),
            rotation_b.coeffs().data(),
            translation_b.data()
        );
    }
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = tolerance_function;
    options.gradient_tolerance = tolerance_gradient;
    options.parameter_tolerance = tolerance_parameter;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void Odometer::bundle_adjustment(std::vector<bool>& to_freeze) {
    if (count_keyframes != this->keyframes.size()) {
        throw std::runtime_error("Wrong number of keyframes present before bundle adjustment.");
    }

    auto problem = ceres::Problem();
    auto loss_function = new ceres::HuberLoss(1.0);

    for (auto& keyframe: keyframes) {
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

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return;
}
