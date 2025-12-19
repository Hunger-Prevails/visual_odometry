#include <algorithm>
#include "image_loader.hpp"


ImageLoader::ImageLoader(const fs::path& folder_path) : current_index(0) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) paths.push_back(entry.path());
    }
    if (paths.empty()) {
        throw std::invalid_argument("no images found under: " + folder_path.string());
    }
    std::sort(paths.begin(), paths.end());
}

bool ImageLoader::hasNext() const {
    return current_index < paths.size();
}

cv::Mat ImageLoader::next() {
    if (!hasNext()) throw std::runtime_error("no more frames to load: " + paths[current_index].string());

    return cv::imread(paths[current_index ++].string(), cv::IMREAD_COLOR);
}

size_t ImageLoader::size() const {
    return paths.size();
}

void ImageLoader::reset() {
    current_index = 0;
}
