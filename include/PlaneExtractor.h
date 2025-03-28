#ifndef RSS_PE_PLANEEXTRACTOR_H
#define RSS_PE_PLANEEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/console/print.h>

#include "ConfigManager.h"

class PlaneExtractor {
private:
    struct SupportPoint {
        int32_t u;
        int32_t v;
        float d;
        Eigen::Vector3d point3D;

        SupportPoint(int32_t u, int32_t v, float d) : u(u), v(v), d(d) {}
    };

    struct Triangle {
        int32_t c1;
        int32_t c2;
        int32_t c3;
        Eigen::Vector3d normal;
        std::vector<int> neighbor;
        int clusterID = -1;

        Triangle(int32_t c1, int32_t c2, int32_t c3) : c1(c1), c2(c2), c3(c3) {}
    };

    struct Edge {
        int32_t v1;
        int32_t v2;

        Edge(int32_t a, int32_t b) {
            if (a < b) {
                v1 = a;
                v2 = b;
            } else {
                v1 = b;
                v2 = a;
            }
        }

        bool operator==(const Edge& other) const {
            return (v1 == other.v1 && v2 == other.v2);
        }
    };

    struct EdgeHash {
        std::size_t operator()(const Edge& e) const {
            auto h1 = std::hash<int32_t>{}(e.v1);
            auto h2 = std::hash<int32_t>{}(e.v2);

            const std::size_t magic = 0x9e3779b97f4a7c15ULL;
            std::size_t result = h1;
            result ^= (h2 + magic + (result << 6) + (result >> 2));

            return result;
        }
    };

    struct Cluster {
        std::vector<int> triIndices;
        Eigen::Vector3d planeNormal;
        double planeDistance = 0.0;
        bool valid = true;
        std::vector<int> neighbors;

        Cluster() : planeNormal(Eigen::Vector3d::Zero()), planeDistance(0.0) {}
    };

    struct Plane {
        Eigen::Vector3d normal;
        double d;
        std::vector<cv::Point2i> planarPointCoords;

        Plane() : normal(Eigen::Vector3d::Zero()), d(0.0) {}
    };

public:
    explicit PlaneExtractor(const ConfigData& config);
    ~PlaneExtractor();

    void runPipeline(cv::Mat& imLeft, cv::Mat& imRight);
    void visualize(std::string option = "plane");
    void resetMembers();

private:
    void correctDistortion(cv::Mat& imLeft, cv::Mat& imRight);

    void performSobelFiltering(cv::Mat& im, uint8_t*& sobelDesc);
    void sobel3x3(const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h);
    void createDescriptor(uint8_t* sobelDesc, uint8_t* I_du, uint8_t* I_dv, int32_t width, int32_t height, int32_t bpl);
    void convolve_cols_3x3(const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h);
    void convolve_101_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h);
    void convolve_121_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h);
    void unpack_8bit_to_16bit(const __m128i a, __m128i& b0, __m128i& b1);
    void pack_16bit_to_8bit_saturate(const __m128i a0, const __m128i a1, __m128i& b);

    void computeMatches();
    inline float computeDisparity(const int32_t& u, const int32_t& v, const bool& right_image);
    void removeInconsistentSupportPoints(float* D_can, int32_t D_can_width, int32_t D_can_height);
    void removeRedundantSupportPoints(float* D_can, int32_t D_can_width, int32_t D_can_height, int32_t redun_max_dist, int32_t redun_threshold, bool vertical);
    inline uint32_t getAddressOffsetImage(const int32_t& u, const int32_t& v, const int32_t& width);
    void applyBilateralFilter();

    void generateMesh();
    void computeNormal(Triangle& t);
    void findNeighbor(std::vector<Triangle>& triangles);
    void discardBadTriangles();
    inline double computeTriAngle(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

    void clusterTriangles();
    PlaneExtractor::Cluster performRegionGrowing(int seedIdx, int clusterIdx);
    void fitPlane(const std::vector<Eigen::Vector3d>& pts3D, Eigen::Vector3d& normal, double& distance);
    void fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector3d& normal, double& distance);
    inline double computeNormalAngle(const Eigen::Vector3d& n1, const Eigen::Vector3d& n2);
    inline double computeDistance(const Triangle& tri, const Eigen::Vector3d& planeNormal, double planeDist);
    inline void collectPoints(const Cluster& cluster, std::vector<Eigen::Vector3d>& pts);
    std::vector<std::vector<int>> extractConnectedComponents(const std::vector<int>& clusterIndices);

    void mergeClusters();
    int getBoundaryTriangleCount(int c1, int c2);
    void merge(int srcIdx, int dstIdx);
    void updateAdjacency();

    void computePlaneParameters();

    std::vector<cv::Scalar> generateColors(int nColors);
    cv::Scalar hsv2bgr(double h, double s, double v);

    ConfigData m_config;

    cv::Mat canvas;

    uint8_t* sobelDescLeft;
    uint8_t* sobelDescRight;

    std::vector<SupportPoint> supportPoints;

    std::vector<Triangle> triangles;

    std::vector<Cluster> clusters;
    std::vector<Cluster> mainClusters;
    std::vector<Plane> planes;
};

#endif //RSS_PE_PLANEEXTRACTOR_H
