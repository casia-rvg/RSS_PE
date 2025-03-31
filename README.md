# RSS: Robust Stereo SLAM with Novel Extraction and Full Exploitation of Plane Features

## Introduction

This repository provides the **open-source plane extraction algorithm** of **RSS**, a novel point-plane-based stereo SLAM system that fully leverages plane features to enhance accuracy and robustness. The algorithm performs **real-time plane extraction** using only a pair of stereo images and can be easily integrated into existing stereo-based SLAM frameworks to introduce plane constraints. For technical details and experimental results, please refer to our [paper](https://ieeexplore.ieee.org/document/10500826).

<div align="center">
    <img src="img/overview.png" width="100%" alt="System Overview" />
</div>

## Prerequisites

The code has been tested on **Ubuntu 18.04** and **Ubuntu 20.04**.

Required libraries:

- **OpenCV** (tested on v3.3.0)
- **Eigen3** (tested on v3.3.4)
- **PCL** (tested on v1.9.0)

Make sure these dependencies are properly installed before building the project.

## Dataset

We have validated our plane extraction algorithm on the following datasets:

- **[EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)**
- **[KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)**

You can download these datasets and provide the stereo images, timestamps, and appropriate configuration files to run the plane extraction module.

## Installation

1. **Clone the repository**:

```bash
git clone git@github.com:casia-rvg/RSS_PE.git
```

2. **Configure and compile**:

```bash
cd RSS_PE
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

If the build completes successfully, the main executable (`rss_pe`) will be generated in the `build` directory.

## Directory Structure

Below is a brief overview of the repository structure after compilation:

```
RSS_PE/
├── build/                # Contains build outputs and the final executable (rss_pe)
├── CMakeLists.txt        # CMake configuration file
├── examples/
│   ├── euroc_timestamps/
│   │   ├── V101.txt      # Example timestamps for the EuRoC V1_01_easy sequence
│   │   └── ...           # Additional timestamp files
│   └── EuRoC.yaml        # Algorithm parameter file for the EuRoC dataset
├── img/                  # Image resource
├── include/              # Header files for core functionalities
├── main.cpp              # Main entry point of the plane extraction module
├── README.md             # Project introduction and usage guide
└── src/                  # Source files implementing the core functionalities
```

## Usage

Run the plane extraction module with:

```bash
./rss_pe <left_image_folder> <right_image_folder> <timestamps_file> <config_yaml>
```

Where:

- `<left_image_folder>`: Path to the directory containing left camera images.
- `<right_image_folder>`: Path to the directory containing right camera images.
- `<timestamps_file>`: Path to the file listing image timestamps (e.g., `V101.txt` for the EuRoC `V1_01_easy` sequence).
- `<config_yaml>`: Path to the YAML file with algorithm parameters (e.g., `EuRoC.yaml` for the EuRoC dataset).

**Example:**

```bash
./rss_pe /path/to/left/images /path/to/right/images /path/to/timestamps.txt /path/to/config.yaml
```

## Citation

If you find this work beneficial for your research or wish to reference it in any publication, please cite it as follows:

```bibtex
@article{wang2024rss,
  title={RSS: Robust Stereo SLAM With Novel Extraction and Full Exploitation of Plane Features},
  author={Wang, Haolin and Wei, Hao and Xu, Zewen and Lv, Zeren and Zhang, Pengju and An, Ning and Tang, Fulin and Wu, Yihong},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Contact

We sincerely hope this project benefits your research and development. If you have any questions, suggestions, or bug reports, please feel free to [open an issue](https://github.com/casia-rvg/RSS_PE/issues) on GitHub.

Thank you for your interest and support!