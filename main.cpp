# include <iostream>
# include <pcl/point_types.h>
# include <pcl/point_cloud.h>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <igl/cotmatrix.h>

int main() {
    pcl::PointCloud<pcl::PointXYZ> cloud;

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

    Eigen::Matrix2i A;

    A << 1, 2,
         3, 4;

    int determinant = A.determinant();

    std::cout << "Eigen Matrix A:\n" << A << std::endl;
    std::cout << "Determinant of A: " << determinant << std::endl;

    Eigen::MatrixXd V(4, 2); // 4 vertices, 2D coordinates (V)
    V << 0, 0,
         1, 0,
         1, 1,
         0, 1;

    Eigen::MatrixXi F(2, 3); // 2 faces (triangles), 3 indices each (F)
    F << 0, 1, 2,
         0, 2, 3;

    Eigen::SparseMatrix<double> L; // The output: a sparse matrix for the Laplacian

    // Use a libigl function!
    igl::cotmatrix(V, F, L);

    std::cout << "Computed Cotangent Laplacian Matrix L:\n" << L << std::endl;

    return 0;
}
