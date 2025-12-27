# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include <vector>

class Extractor {
    cv::Ptr<cv::SIFT> sift;

public:
    Extractor(int count_features);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> extract(cv::Mat& image);
};
