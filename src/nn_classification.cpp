/*
 * NN_classification.cpp
 * Author: Huili Yu
 */
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../include/neural_network.h"

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cout << "Please specify an option "
    "(data_generation, training, or test)." << std::endl;
    std::cout << "Please also specify a data file." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string option = argv[1];
  neural_network::NeuralNet nn;
  std::string data_file = argv[2];

  if (option == "data_generation") {
    if (argc < 4) {
      std::cout << "Please image directories." << std::endl;
      exit(EXIT_FAILURE);  
    }
    // Load images and write to data files
    std::vector<std::string> directories;
    for (int i = 3; i < argc; ++i) {
      directories.push_back(argv[i]);
    }
    std::vector<std::string> files;
    nn.ReadFiles(directories, files);
    nn.WriteDataToFile(files, data_file);
  }
  else if (option == "training" || option == "test") {
    if (argc < 4) {
      std::cout << "Please specify a model file for "
      "training or test options." << std::endl;
      exit(EXIT_FAILURE);
    }
    std::string model_file = argv[3];
    // Load data file
    nn.LoadDataFromFiles(data_file);
    if (option == "training") {
      nn.Print();
      // Train neural network
      nn.Train(model_file);
    }
    nn.Predict(model_file);
  }

  std::cout << "End of main." << std::endl;
  return 0;
}
