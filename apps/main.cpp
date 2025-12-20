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


Eigen::Matrix3f load_intrinsics(const fs::path& filepath, const std::string& camera) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::invalid_argument("Failed to open file: " + filepath.string());
    }

    json matrix_data;
    file >> matrix_data;
    file.close();

    Eigen::Matrix3f matrix;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix(i, j) = matrix_data[camera][i][j].get<float>();
        }
    }
    return matrix;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("visual-odometry", "options to configure visual odometry pipeline");

    options.add_options()("sequence", "Path to the image sequence to process", cxxopts::value<fs::path>());
    options.add_options()("camera", "Name of camera", cxxopts::value<std::string>()->default_value("default"));
    options.add_options()("temporal_baseline", "Number of frames between the two frames chosen for initialization", cxxopts::value<int>()->default_value("10"));

    auto args = options.parse(argc, argv);

    std::cout << "to process sequence " << args["sequence"].as<fs::path>() << std::endl;

    std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>(args["sequence"].as<fs::path>() / "rgb");

    auto intrinsics_path = fs::canonical(argv[0]).parent_path() / "res/intrinsics.json";

    Eigen::Matrix3f intrinsics = load_intrinsics(intrinsics_path, args["camera"].as<std::string>());

    std::cout << "intrinsics matrix:\n" << intrinsics << std::endl;

    auto odometer = std::make_unique<Odometer>(intrinsics, loader, args["temporal_baseline"].as<int>());

    std::cout << "to start visual odometry" << std::endl;

    odometer->initialize();
    odometer->processFrames();

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
