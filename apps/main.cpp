# include <iostream>
# include <cxxopts.hpp>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>

int main(int argc, char* argv[]) {
    cxxopts::Options options("visual-odometry", "options to configure visual odometry pipeline");

	options.add_options()("sequence", "Path to the image sequence to process", cxxopts::value<std::string>());

    auto args = options.parse(argc, argv);

    std::cout << args["sequence"].as<std::string>() << std::endl;

    return 0;
}
