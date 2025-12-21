# include <map>
# include <filesystem>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class ImageLoader;

class Odometer {
protected:
    bool is_initialized;
    int temporal_baseline;
    int count_features;
    fs::path write_path;

    Eigen::Matrix3f intrinsics;
    std::shared_ptr<ImageLoader> loader;

    std::map<int, Eigen::Quaternionf> rotations;
    std::map<int, Eigen::Vector3f> translations;

public:
    Odometer(Eigen::Matrix3f intrinsics, std::shared_ptr<ImageLoader> loader, fs::path write_path, int temporal_baseline = 10, int count_features = 2000);

    void initialize();
    void processFrame(int index);
    void processFrames();

    const std::vector<Eigen::Quaternionf> getRotations();
    const std::vector<Eigen::Vector3f> getTranslations();
};
