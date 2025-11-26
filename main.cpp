#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

int main() {
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Fill the cloud with some dummy data
    cloud.width = 5;
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);

    for (auto& point : cloud) {
        point.x = 1024 * rand() / (RAND_MAX + 1.0f);
        point.y = 1024 * rand() / (RAND_MAX + 1.0f);
        point.z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    std::cout << "Generated " << cloud.size() << " points." << std::endl;
    for (const auto& point : cloud) {
        std::cout << "    " << point.x << " " << point.y << " " << point.z << std::endl;
    }
    std::cout << "PCL point cloud example placeholder." << std::endl;

    return 0;
}