# include <iostream>
# include <pcl/point_types.h>
# include <pcl/point_cloud.h>
# include <ceres/ceres.h>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <igl/cotmatrix.h>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>


struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

int main() {
    std::cout << "--- PCL Test ---" << std::endl;
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
    std::cout << std::endl;

    std::cout << "--- Ceres Test ---" << std::endl;
    double x = 0.5;
    double initial_x = x;

    ceres::Problem problem;
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << "Initial x: " << initial_x << " -> Final x: " << x << std::endl;
    std::cout << "Success: " << (std::abs(x - 10.0) < 1e-6 ? "YES" : "NO") << std::endl;
    std::cout << summary.FullReport() << std::endl;
    std::cout << std::endl;

    std::cout << "--- Eigen Test ---" << std::endl;
    Eigen::Matrix2i A;

    A << 1, 2,
         3, 4;

    int determinant = A.determinant();

    std::cout << "Eigen Matrix A:\n" << A << std::endl;
    std::cout << "Determinant of A: " << determinant << std::endl;
    std::cout << std::endl;

    std::cout << "--- libigl Test ---" << std::endl;
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
    std::cout << std::endl;

    std::cout << "--- OpenCV Test ---" << std::endl;
    // Check SIFT
    try {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        std::cout << "SIFT Detector: Successfully initialized." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "SIFT Error: " << e.what() << std::endl;
    }

    // Check Eigen Integration
    Eigen::Matrix3d eigen_mat = Eigen::Matrix3d::Identity();
    cv::Mat cv_mat;
    cv::eigen2cv(eigen_mat, cv_mat);

    if (cv_mat.at<double>(0,0) == 1.0 && cv_mat.rows == 3) {
        std::cout << "Eigen -> OpenCV Bridge: Working correctly." << std::endl;
    }

    // Test Image Container
    cv::Mat test_img = cv::Mat::zeros(100, 100, CV_8UC1);
    if (!test_img.empty()) {
        std::cout << "Core Image Container: Working correctly." << std::endl;
    }
    return 0;
}
