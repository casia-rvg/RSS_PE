#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "ConfigManager.h"
#include "PlaneExtractor.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <left_image_folder> <right_image_folder>"
                  << " <timestamp_txt> <parameter_yaml>\n";
        return -1;
    }

    std::string leftFolder = argv[1];
    std::string rightFolder = argv[2];
    std::string timestampFile = argv[3];
    std::string parameterFile = argv[4];

    ConfigManager configManager(parameterFile);
    if (!configManager.load()) {
        std::cerr << "Failed to load config from: " << parameterFile << std::endl;
        return -1;
    }

    std::ifstream timestampFin(timestampFile);
    if (!timestampFin.is_open()) {
        std::cerr << "Failed to open timestamp file: " << timestampFile << std::endl;
        return -1;
    }

    PlaneExtractor pe(configManager.getConfigData());

    std::string line;
    while (getline(timestampFin, line)) {
        std::string image_name = line + ".png";
        std::string left_path = leftFolder + "/" + image_name;
        std::string right_path = rightFolder + "/" + image_name;

        cv::Mat imLeft = imread(left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat imRight = imread(right_path, cv::IMREAD_GRAYSCALE);

        pe.runPipeline(imLeft, imRight);
        pe.visualize();
        pe.resetMembers();
    }

    return 0;
}
