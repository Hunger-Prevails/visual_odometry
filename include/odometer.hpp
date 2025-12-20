# include <map>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>


class ImageLoader;

class Odometer {
protected:
    bool is_initialized;
    int temporal_baseline;
    Eigen::Matrix3f intrinsics;
    std::shared_ptr<ImageLoader> loader;

    std::map<int, Eigen::Quaternionf> rotations;
    std::map<int, Eigen::Vector3f> translations;

public:
    Odometer(Eigen::Matrix3f intrinsics, std::shared_ptr<ImageLoader> loader, int temporal_baseline = 10);

    void initialize();
    void processFrame(int index);
    void processFrames();

    const std::vector<Eigen::Quaternionf> getRotations();
    const std::vector<Eigen::Vector3f> getTranslations();
};
