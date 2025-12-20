#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace fs = std::filesystem;

class ImageLoader {
private:
    std::vector<fs::path> paths;

public:
    ImageLoader(const fs::path& folder_path);

    cv::Mat operator[](size_t index) const;
    size_t size() const;
};

#endif // IMAGE_LOADER_HPP
