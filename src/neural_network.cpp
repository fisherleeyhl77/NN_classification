/*
 * neural_network.cpp
 * Author: Huili Yu
 */

// Header
#include "../include/neural_network.h"
#include <dirent.h>
#include <assert.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <iterator>

namespace neural_network
{
// Reads a vector of file directories and
// outputs a vector of filename strings
void NeuralNet::ReadFiles(const std::vector<std::string> &directories,
                          std::vector<std::string> &files)
{
  int n_directs = (int)directories.size();
  if (n_directs == 0) {
    std::cerr << "The size of input directories is zero" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n_directs; ++i) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directories[i].c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 ||
          strcmp(ent->d_name, "..") == 0) {
          continue;
        }
        files.push_back(directories[i] + std::string(ent->d_name));
      }
      n_samples_classes_.push_back((int)files.size());
      closedir (dir);
    } else {
      /* could not open directory */
      std::cerr << "Failed to open the directory" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

// Generate the data file for the images
void NeuralNet::WriteDataToFile(std::vector<std::string> &files,
                                std::string &file_name)
{
  int n = (int)files.size();
  if (n <= 0) {
    std::cerr << "There is no data" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ofstream output_file_pos(file_name.c_str());
  if (output_file_pos.is_open()) {
    int class_idx = 0;
    for (int i = 0; i < n; ++i) {
        cv::Mat img = cv::imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            std::cout << "Failed to read image: " << files[i] << std::endl;
            exit(EXIT_FAILURE);
        }
        std::vector<float> tmp_data(img.rows * img.cols, 0.0f);
        for (int k = 0; k < img.rows; ++k) {
          for (int j = 0; j < img.cols; ++j) {
            tmp_data[k * img.cols + j] = (float)img.at<uchar>(k, j) / 256.0f;
          }
        }

        if (i == 0) {
          output_file_pos << n << std::endl;
          output_file_pos << img.rows << "," << img.cols << std::endl;
          output_file_pos << (int)n_samples_classes_.size() << std::endl;
        }

        std::ostringstream oss;
        std::copy(tmp_data.begin(), tmp_data.end()-1,
                  std::ostream_iterator<float>(oss, ","));
        oss << tmp_data.back();
        if ((i + 1) > n_samples_classes_[class_idx]) {
          ++class_idx;
        }
        std::string tmp = "," + std::to_string(class_idx);
        output_file_pos << oss.str() << tmp << std::endl;
    }
    output_file_pos.close();
  }
  else {
    std::cerr << "Failed to open the output file." << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Loads data file into a Mat structure
void NeuralNet::LoadFile(const std::string &file_name, cv::Mat &data)
{
  std::ifstream ifs(file_name.c_str());
  std::string line, num;
  if (ifs.is_open()) {
    int counter = 0;
    while (std::getline(ifs, line)) {
      ++counter;
      if (counter < 4) {
        continue;
      }

      int col_idx = 0;
      std::stringstream ss(line);
      while (std::getline(ss, num, ',')) {
        data.at<float>(counter - 4, col_idx) = atof(num.c_str());
        ++col_idx;
      }
    }
    ifs.close();
  }
  else {
    std::cerr << "Failed to open the data file." << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Load data from the data file
void NeuralNet::LoadDataFromFiles(const std::string &data_file)
{
  std::ifstream ifs(data_file.c_str());
  std::string line, num;
  if (ifs.is_open()) {
    int counter = 0;
    while (std::getline(ifs, line)) {
      ++counter;
      // Get the number of samples
      if (counter == 1) {
        n_samples_ = atoi(line.c_str());
        continue;
      }

      std::stringstream ss(line);
      // Get the number of features
      if (counter == 2) {
        std::getline(ss, num, ',');
        int img_height = atoi(num.c_str());
        std::getline(ss, num, ',');
        int img_width = atoi(num.c_str());
        n_features_ = img_height * img_width;
        continue;
      }
      // Get the number of classes
      if (counter == 3) {
        n_classes_ = atoi(line.c_str());
        break;
      }
    }
  }
  else {
    std::cerr << "Failed to open the data file." << std::endl;
    exit(EXIT_FAILURE);
  }

  data_ = cv::Mat(n_samples_, n_features_ + 1, CV_32FC1);
  LoadFile(data_file, data_);

  // Create one-hot output based on labels
  labels_one_hot_ =
      cv::Mat(data_.rows, n_classes_, CV_32FC1);
  for (int i = 0; i < labels_one_hot_.rows; ++i) {
    labels_one_hot_.row(i).setTo(cv::Scalar(0.0f));
    int j = data_.at<float>(i, data_.cols - 1);
    labels_one_hot_.at<float> (i, j) = 1.0f;
  }

  labels_ = data_.col(data_.cols - 1);
  data_ = data_.colRange(0, data_.cols - 1);
}

// Prints the information about the data of
// neural network
void NeuralNet::Print()
{
  std::cout << (int)data_.rows << " " <<
      (int)data_.cols << std::endl;
  std::cout << (int)labels_one_hot_.rows << " " <<
      (int)labels_one_hot_.cols << std::endl;
}

// Train the neural network
void NeuralNet::Train(const std::string &model_file)
{
  // Set up the neural network
  cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
  layers.row(0) = cv::Scalar(n_features_);      // input
  layers.row(1) = cv::Scalar(n_classes_ * 8);   // hidden
  layers.row(2) = cv::Scalar(n_classes_ * 4);   // hidden
  layers.row(3) = cv::Scalar(n_classes_);       // output

  CvANN_MLP_TrainParams params;
  CvTermCriteria criteria;
  criteria.max_iter = 300;
  criteria.epsilon = 0.0001f;
  criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

  params.train_method = CvANN_MLP_TrainParams::BACKPROP;
  params.bp_dw_scale = 0.05f;
  params.bp_moment_scale = 0.05f;
  params.term_crit = criteria;

  // Create the ANN
  ann_.create(layers);

  // Train
  std::cout << "Training started" << std::endl;
  ann_.train(data_, labels_one_hot_,
            cv::Mat(), cv::Mat(), params);
  std::cout << "Training finished" << std::endl;

  // Store the trained model
  cv::FileStorage fs(model_file, cv::FileStorage::WRITE);
  ann_.write(*fs, "mlp");
}

// Prediction
void NeuralNet::Predict(const std::string &model_file)
{
  // If Model exits, then load the neural network model
  // for evaluation using test data
  if (!model_file.empty()) {
    ann_.load(model_file.c_str(), "mlp");
  }

  // Otherwise, evaluate the model just trained using
  // training data
  cv::Mat predicted(labels_.rows, 1, CV_32F);
  for (int i = 0; i < data_.rows; ++i) {
    cv::Mat response(1, n_classes_, CV_32FC1);
    cv::Mat sample = data_.row(i);
    ann_.predict(sample, response);
    double max_val = DBL_MIN;
    int max_idx = INT_MIN;
    for (int j = 0; j < n_classes_; ++j) {
      if (max_val < response.at<float>(0, j)) {
        max_val = response.at<float>(0, j);
        max_idx = j;
      }
    }
    predicted.at<float>(i, 0) = max_idx;
  }

  // Evaluation the model
  std::cout << "Accuracy_{MLP} = " <<
  Evaluate(predicted, labels_) << std::endl;
}

// Evaluate the model by computing accuracy
float NeuralNet::Evaluate(cv::Mat& predicted, cv::Mat& actual)
{
  assert(predicted.rows == actual.rows);
  int n_corrects = 0;
  for (int i = 0; i < predicted.rows; ++i) {
    if (predicted.at<float>(i, 0) == actual.at<float>(i,0)) {
      ++n_corrects;
    }
  }
  return n_corrects * 1.0f / predicted.rows;
}
}  // namespace neural_network
