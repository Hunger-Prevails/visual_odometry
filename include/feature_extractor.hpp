# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include <vector>

class Extractor {
    cv::Ptr<cv::SIFT> sift;

public:
    Extractor(int count_features);
    void extract(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};
