#ifndef Model_H
#define Model_H
#define nbB 4

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

class Model {
  //   struct FaceBox {
  //     float x;
  //     float y;
  //     float w;
  //     float h;
  //   };

  //   struct FaceRes {
  //     float confidence;
  //     FaceBox face_box;
  //     std::vector<cv::Point2f> keypoints;
  //   };

public:
  explicit Model(const std::string &config_file);
  ~Model();
  void LoadEngine();
  void Destroy();
  std::vector<std::vector<float *>> EngineInference(const std::vector<cv::Mat> &image_list);
  int nbBindings = nbB;
  void *buffers[nbB]; // nbBindings
  int outSize[nbB-1];   // nbBindings-1

private:
  std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img);
  //   std::vector<std::vector<FaceRes>>
  //   postProcess(const std::vector<cv::Mat> &vec_Mat, float *output_1,
  //               float *output_2, float *output_3, float *output_4,
  //               const int &outSize_1, const int &outSize_2, const int
  //               &outSize_3, const int &outSize_4);

  std::string onnx_file;
  std::string engine_file;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;

  
  cudaStream_t stream;
  std::vector<int64_t> bufferSize;
  int BATCH_SIZE;
  int INPUT_CHANNEL;
  int IMAGE_WIDTH;
  int IMAGE_HEIGHT;
};

#endif // Model_Model_H
