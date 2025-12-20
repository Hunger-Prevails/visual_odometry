#include <algorithm>
#include "image_loader.hpp"


ImageLoader::ImageLoader(const fs::path& folder_path) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) paths.push_back(entry.path());
    }
    if (paths.empty()) {
        throw std::invalid_argument("no images found under: " + folder_path.string());
    }
    std::sort(paths.begin(), paths.end());
}

cv::Mat ImageLoader::operator[](size_t index) const {
    if (index >= paths.size()) {
        throw std::out_of_range("Index out of range in ImageLoader");
    }
    return cv::imread(paths[index].string(), cv::IMREAD_COLOR);
}

size_t ImageLoader::size() const {
    return paths.size();
}
