/*
 * neural_network.h
 * A full connected neural network training and testing using OpenCV
 * Author: Huili Yu
 */

#ifndef INCLUDES_NEURAL_NETWORK_H_
#define INCLUDES_NEURAL_NETWORK_H_

// Headers
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace neural_network
{
class NeuralNet
{
 public:
  NeuralNet() {}

  /*
   * ReadFiles function reads a vector of file directories
   * and outputs a vector of filename strings
   * @param directories: where the training or test images are
   * @param files: a vector of filename strings
   */
  void ReadFiles(const std::vector<std::string> &directories,
                 std::vector<std::string> &files);

  /*
   * WriteDataToFile function generates the data file for the images
   * @param files: a vector of filename strings
   * @file_name: a string specifying where the data file will be
   */
  void WriteDataToFile(std::vector<std::string> &files,
                       std::string &file_name);

  /*
   * LoadFile function loads data file into a Mat structure
   * @param file_name: a string specifying the path of data file
   * @param data: a Mat structure storing data
   */
  void LoadFile(const std::string &file_name, cv::Mat &data);

  /*
   * LoadDataFromFiles function load data from the data file
   * @param data_file: a string specifying the path of the data file
   */
  void LoadDataFromFiles(const std::string &data_file);

  /*
   * Print function prints the information about the data of
   * neural network
   */
  void Print();

  /*
   * Train function trains the neural network model using the data
   * and store the model to a model file
   * @param model_file: the path of the file where the model
   * will be stored
   */
  void Train(const std::string &model_file);

  /*
   * Predict function predicts for new data and evaluate
   * the prediction performance
   * @param model_file: the path of the file where the model is stored
   */
  void Predict(const std::string &model_file);

  /*
   * Evaluate functions evaluates the prediction accuracy
   * @param predicted: stores the predictions
   * @param actual: stores the ground truth
   */
  float Evaluate(cv::Mat& predicted, cv::Mat& actual);
 private:
  int n_classes_;        // number of classes
  int n_samples_;        // number of samples
  int n_features_;       // number of features
  cv::Mat data_;
  cv::Mat labels_one_hot_;
  cv::Mat labels_;
  std::vector<int> n_samples_classes_;  // number of samples up to a class
  CvANN_MLP ann_;        // neural network
};
}  // namespace neural network

#endif /* INCLUDES_NEURAL_NETWORK_H_ */
