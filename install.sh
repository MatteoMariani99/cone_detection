#!/bin/bash
sudo apt update
read -p "Create new conda env (y/n)?" CONT

if [ "$CONT" == "n" ]; then
  echo "exit";
else
  # user chooses to create conda env
  # prompt user for conda env name
  read -p "Creating new conda environment, choose name: " input_variable
  echo "Name $input_variable was chosen";


  #list name of packages
  
  echo "installing base packages"
  conda create --name $input_variable\
  python=3.11 
  echo "Conda environment create complete"
  echo "Install other dependencies"
  pip install ultralytics
  echo "Ultralytics installation complete"
  wget -O fsoco_dataset https://universe.roboflow.com/ds/9lTdQO9Ufw?key=P2faPZEQ3e
  echo "Downlod dataset complete"
  echo "Unzip dataset"
  mkdir fsoco_dataset_yolov8
  unzip fsoco_dataset -d fsoco_dataset_yolov8
fi




