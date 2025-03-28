#include "ConfigManager.h"

ConfigManager::ConfigManager(const std::string& yamlPath)
        : m_yamlPath(yamlPath)
{

}

bool ConfigManager::load() {
    cv::FileStorage fs(m_yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ConfigManager] Could not open file: " << m_yamlPath << std::endl;
        return false;
    }

    fs["width"] >> m_configData.width;
    fs["height"] >> m_configData.height;

    fs["LEFT.K"] >> m_configData.K_l;
    fs["LEFT.D"] >> m_configData.D_l;
    fs["LEFT.R"] >> m_configData.R_l;
    fs["LEFT.P"] >> m_configData.P_l;

    fs["RIGHT.K"] >> m_configData.K_r;
    fs["RIGHT.D"] >> m_configData.D_r;
    fs["RIGHT.R"] >> m_configData.R_r;
    fs["RIGHT.P"] >> m_configData.P_r;

    cv::initUndistortRectifyMap(
            m_configData.K_l, m_configData.D_l, m_configData.R_l,
            m_configData.P_l(cv::Range(0, 3), cv::Range(0, 3)),
            cv::Size(m_configData.width, m_configData.height), CV_32F,
            m_configData.M1l, m_configData.M2l
    );

    cv::initUndistortRectifyMap(
            m_configData.K_r, m_configData.D_r, m_configData.R_r,
            m_configData.P_r(cv::Range(0, 3), cv::Range(0, 3)),
            cv::Size(m_configData.width, m_configData.height), CV_32F,
            m_configData.M1r, m_configData.M2r
    );

    fs["cx"] >> m_configData.cx;
    fs["cy"] >> m_configData.cy;
    fs["fx"] >> m_configData.fx;
    fs["fy"] >> m_configData.fy;
    fs["bf"] >> m_configData.bf;

    fs["candidateStepsize"] >> m_configData.candidateStepsize;
    fs["lrThreshold"] >> m_configData.lrThreshold;
    fs["supportTexture"] >> m_configData.supportTexture;
    fs["dispMin"] >> m_configData.dispMin;
    fs["dispMax"] >> m_configData.dispMax;
    fs["supportThreshold"] >> m_configData.supportThreshold;
    fs["inconWindowSize"] >> m_configData.inconWindowSize;
    fs["inconThreshold"] >> m_configData.inconThreshold;
    fs["inconMinSupport"] >> m_configData.inconMinSupport;

    fs["sigDisp"] >> m_configData.sigDisp;
    fs["sigDist"] >> m_configData.sigDist;
    fs["filterRadius"] >> m_configData.filterRadius;

    fs["maxValidEdge"] >> m_configData.maxValidEdge;
    fs["checkOutlier"] >> m_configData.checkOutlier;
    fs["badEdgeThresh"] >> m_configData.badEdgeThresh;
    fs["badAspectThresh"] >> m_configData.badAspectThresh;
    fs["badAngleThresh"] >> m_configData.badAngleThresh;
    fs["checkNN"] >> m_configData.checkNN;
    fs["boundaryThresh"] >> m_configData.boundaryThresh;

    fs["RansacDistance"] >> m_configData.RansacDistance;
    fs["angleThreshWide"] >> m_configData.angleThreshWide;
    fs["distThreshWide"] >> m_configData.distThreshWide;
    fs["reRansacBatch"] >> m_configData.reRansacBatch;
    fs["angleThreshStrict"] >> m_configData.angleThreshStrict;
    fs["distThreshStrict"] >> m_configData.distThreshStrict;
    fs["angleThreshMerge"] >> m_configData.angleThreshMerge;
    fs["distThreshMerge"] >> m_configData.distThreshMerge;

    fs["minClusterRatio"] >> m_configData.minClusterRatio;
    fs["inlierRatio"] >> m_configData.inlierRatio;
    fs["angleThreshDupPlane"] >> m_configData.angleThreshDupPlane;
    fs["distThreshDupPlane"] >> m_configData.distThreshDupPlane;

    fs.release();
    return true;
}

const ConfigData& ConfigManager::getConfigData() const {
    return m_configData;
}
