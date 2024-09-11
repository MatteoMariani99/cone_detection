#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    echo "Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
else
    echo "Conda is already installed."
fi

  echo "installing base packages"
  conda env create -f cone_detection_yolov8.yml
  echo "Conda environment create complete"
  echo "Download dataset"
  wget -O fsoco_dataset https://universe.roboflow.com/ds/9lTdQO9Ufw?key=P2faPZEQ3e
  echo "Download dataset complete"
  echo "Unzip dataset"
  mkdir fsoco_dataset_yolov8
  unzip fsoco_dataset -d fsoco_dataset_yolov8
  rm -rf fsoco_dataset

fi




