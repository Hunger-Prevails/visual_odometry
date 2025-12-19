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
    size_t current_index;

public:
    ImageLoader(const fs::path& folder_path);

    bool hasNext() const;
    cv::Mat next();
    size_t size() const;
    void reset();
};

#endif // IMAGE_LOADER_HPP
