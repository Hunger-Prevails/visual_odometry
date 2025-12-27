# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include <vector>
# include <unordered_map>
# include "feature_matcher.hpp"

const int Matcher::n_neighbors(2);

Matcher::Matcher(float test_ratio): test_ratio(test_ratio) {
    matcher = cv::FlannBasedMatcher::create();
}

std::vector<cv::DMatch> Matcher::match(cv::Mat& descriptors_a, cv::Mat& descriptors_b) {
    std::vector<cv::DMatch> matches;

    if (descriptors_a.type() != CV_32F) {
        descriptors_a.convertTo(descriptors_a, CV_32F);
    }
    if (descriptors_b.type() != CV_32F) {
        descriptors_b.convertTo(descriptors_b, CV_32F);
    }
    matcher->match(descriptors_a, descriptors_b, matches);

    return matches;
}

std::vector<cv::DMatch> Matcher::enforce_bijection(const std::vector<cv::DMatch>& matches) {
    std::unordered_map<int, int> registry;
    std::vector<cv::DMatch> matches_bijective;

    for (int i = 0; i < matches.size(); i ++) {
        int t_idx = matches[i].trainIdx;

        if (registry.find(t_idx) == registry.end()) {
            registry.emplace(t_idx, i);
        }
        else if (matches[i].distance < matches[registry[t_idx]].distance) {
            registry[t_idx] = i;
        }
    }

    for (auto [t_idx, match_idx]: registry) {
        matches_bijective.push_back(matches[match_idx]);
    }
    return matches_bijective;
}

std::vector<cv::DMatch> Matcher::match_knn(cv::Mat& descriptors_a, cv::Mat& descriptors_b) {
    std::vector<std::vector<cv::DMatch>> matches_knn;

    if (descriptors_a.type() != CV_32F) {
        descriptors_a.convertTo(descriptors_a, CV_32F);
    }
    if (descriptors_b.type() != CV_32F) {
        descriptors_b.convertTo(descriptors_b, CV_32F);
    }
    matcher->knnMatch(descriptors_a, descriptors_b, matches_knn, Matcher::n_neighbors);

    std::cout << "Found " << matches_knn.size() << " initial matches" << std::endl;

    std::vector<cv::DMatch> matches;

    for (size_t i = 0; i < matches_knn.size(); i++) {
        if (matches_knn[i].size() == Matcher::n_neighbors) {
            float d1 = matches_knn[i][0].distance;
            float d2 = matches_knn[i][1].distance;

            if (d1 / d2 < test_ratio) {
                matches.push_back(matches_knn[i][0]);
            }
        }
    }
    std::cout << "To keep " << matches.size() << " matches after ratio test" << std::endl;

    auto matches_bijective = enforce_bijection(matches);

    std::cout << "To keep " << matches_bijective.size() << " bijective matches after bijection enforcement" << std::endl;

    return matches_bijective;
}
