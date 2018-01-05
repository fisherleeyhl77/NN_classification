# Introduction
The code is to generate training or test data from images, and then train and test a fully connected neural network for object classification based on the data using OpenCV. 

# Requirements
* OpenCV 2.X
* CMake > 2.8

# Usage
* mkdir bin
* mkdir build
* mkdir data
* mkdir models
* cd build
* cmake ..
* make
* cd ..
* For data generation, run ./bin/nn_classification data_generation ./data/data_file_name.txt ./data/positive_image_sample_directories ./data/negative_image_sample_directories
* For training, run ./bin/nn_classification training ./data/data_file_name.txt ./models/model_file_name.yml
* For test, run ./bin/nn_classification test ./data/data_file_name.txt ./models/model_file_name.yml
