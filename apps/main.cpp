# include <iostream>
# include <cxxopts.hpp>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>
# include <nlohmann/json.hpp>
# include <fstream>
# include <stdexcept>
# include "image_loader.hpp"
# include "odometer.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;


Eigen::Matrix3d load_intrinsics(const fs::path& filepath, const std::string& camera) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::invalid_argument("Failed to open file: " + filepath.string());
    }

    json matrix_data;
    file >> matrix_data;
    file.close();

    Eigen::Matrix3d matrix;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix(i, j) = matrix_data[camera][i][j].get<double>();
        }
    }
    return matrix;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("visual-odometry", "options to configure visual odometry pipeline");

    options.add_options()("sequence", "Path to the image sequence to process", cxxopts::value<fs::path>());
    options.add_options()("write_path", "Path to write outputs to", cxxopts::value<fs::path>()->default_value("outputs"));
    options.add_options()("camera", "Name of camera", cxxopts::value<std::string>()->default_value("default"));
    options.add_options()("temporal_baseline", "Number of frames between the two frames chosen for initialization", cxxopts::value<int>()->default_value("10"));
    options.add_options()("n_keyframes", "Number of keyframes to maintain in memory", cxxopts::value<int>()->default_value("2"));
    options.add_options()("count_features", "Maximum numbers of features to detect on a frame", cxxopts::value<int>()->default_value("2000"));
    options.add_options()("test_ratio", "Test ratio against which to filter matches", cxxopts::value<float>()->default_value("0.75"));
    options.add_options()("track_ratio", "Ratio of existant landmarks to track for non-keyframes", cxxopts::value<float>()->default_value("0.6"));
    options.add_options()("function_tolerance", "Function tolerance for bundle adjustment", cxxopts::value<float>()->default_value("1e-4"));

    auto args = options.parse(argc, argv);

    std::cout << "to process sequence " << args["sequence"].as<fs::path>() << std::endl;

    std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>(args["sequence"].as<fs::path>() / "rgb");

    auto intrinsics_path = fs::canonical(argv[0]).parent_path() / "../res/intrinsics.json";
    auto write_path = fs::canonical(argv[0]).parent_path() / args["write_path"].as<fs::path>();

    auto intrinsics = load_intrinsics(intrinsics_path, args["camera"].as<std::string>());

    std::cout << "to assume intrinsics matrix:\n" << intrinsics << std::endl;

    auto odometer = std::make_unique<Odometer>(
        intrinsics,
        loader,
        write_path,
        args["temporal_baseline"].as<int>(),
        args["n_keyframes"].as<int>(),
        args["count_features"].as<int>(),
        args["test_ratio"].as<float>(),
        args["track_ratio"].as<float>(),
        args["function_tolerance"].as<float>()
    );

    std::cout << "to start visual odometry" << std::endl;

    odometer->initialize();
    odometer->process_frames();

    auto rotations = odometer->getRotations();
    auto translations = odometer->getTranslations();

    if (rotations.size() != translations.size()) {
        throw std::runtime_error("Mismatch between number of rotations and translations.");
    }
    if (rotations.size() != loader->size()) {
        throw std::runtime_error("Mismatch between number of poses and number of images.");
    }
    return 0;
}
