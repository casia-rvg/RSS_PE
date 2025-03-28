#ifndef RSS_PE_CONFIGMANAGER_H
#define RSS_PE_CONFIGMANAGER_H

#include <opencv2/opencv.hpp>
#include <string>

struct ConfigData {
    int32_t width;
    int32_t height;

    cv::Mat K_l;
    cv::Mat D_l;
    cv::Mat R_l;
    cv::Mat P_l;

    cv::Mat K_r;
    cv::Mat D_r;
    cv::Mat R_r;
    cv::Mat P_r;

    cv::Mat M1l;
    cv::Mat M2l;
    cv::Mat M1r;
    cv::Mat M2r;

    double cx;
    double cy;
    double fx;
    double fy;
    double bf;

    int32_t candidateStepsize;
    int32_t lrThreshold;
    int32_t supportTexture;
    int32_t dispMin;
    int32_t dispMax;
    float supportThreshold;
    int32_t inconWindowSize;
    int32_t inconThreshold;
    int32_t inconMinSupport;

    float sigDisp;
    float sigDist;
    int filterRadius;

    double maxValidEdge;
    bool checkOutlier;
    double badEdgeThresh;
    double badAspectThresh;
    double badAngleThresh;
    bool checkNN;
    double boundaryThresh;

    float RansacDistance;
    int angleThreshWide;
    double distThreshWide;
    int reRansacBatch;
    int angleThreshStrict;
    double distThreshStrict;
    int angleThreshMerge;
    double distThreshMerge;

    float minClusterRatio;
    float inlierRatio;
    int angleThreshDupPlane;
    double distThreshDupPlane;

    ConfigData() = default;
};

class ConfigManager {
public:
    explicit ConfigManager(const std::string& yamlPath);

    bool load();
    const ConfigData& getConfigData() const;

private:
    std::string m_yamlPath;
    ConfigData  m_configData;
};

#endif //RSS_PE_CONFIGMANAGER_H
