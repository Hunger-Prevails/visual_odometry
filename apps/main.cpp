# include <iostream>
# include <cxxopts.hpp>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>
# include "image_loader.hpp"
# include "odometer.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    cxxopts::Options options("visual-odometry", "options to configure visual odometry pipeline");

    options.add_options()("sequence", "Path to the image sequence to process", cxxopts::value<fs::path>());
    options.add_options()("folder", "Name of folder to load images from", cxxopts::value<std::string>()->default_value("rgb"));

    auto args = options.parse(argc, argv);

    std::cout << "to process sequence " << args["sequence"].as<fs::path>() << std::endl;

    std::unique_ptr<ImageLoader> loader = std::make_unique<ImageLoader>(args["sequence"].as<fs::path>() / args["folder"].as<std::string>());

    while (loader->hasNext())
    {
        auto image = loader->next();
        std::cout << "Loaded image of size: " << image.cols << " x " << image.rows << std::endl;
    }
    return 0;
}
