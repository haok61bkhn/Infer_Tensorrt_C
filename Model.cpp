#include "Model.h"
#include "common.hpp"
#include "yaml-cpp/yaml.h"

Model::Model(const std::string &config_file) {
  std::cout << config_file;
  YAML::Node root = YAML::LoadFile(config_file);
  YAML::Node config = root["Model"];
  onnx_file = config["onnx_file"].as<std::string>();
  engine_file = config["engine_file"].as<std::string>();
  BATCH_SIZE = config["BATCH_SIZE"].as<int>();
  INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
  IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
  IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
}

Model::~Model() = default;

void Model::LoadEngine() {
  // create and load engine
  std::fstream existEngine;
  existEngine.open(engine_file, std::ios::in);
  if (existEngine) {
    readTrtFile(engine_file, engine);
    assert(engine != nullptr);
  } else {
    onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
    assert(engine != nullptr);
  }

  // get context
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);

  // get buffers
  assert(engine->getNbBindings() == nbBindings);
  bufferSize.resize(nbBindings);

  for (int i = 0; i < nbBindings; ++i) {
    nvinfer1::Dims dims = engine->getBindingDimensions(i);
    nvinfer1::DataType dtype = engine->getBindingDataType(i);
    int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
    bufferSize[i] = totalSize;
    // std::cout << "binding" << i << ": " << totalSize << std::endl;
    cudaMalloc(&buffers[i], totalSize);
  }

  // get stream

  cudaStreamCreate(&stream);
  for (int i = 1; i < nbBindings; i++)
    outSize[i - 1] = int(bufferSize[i] / sizeof(float) / BATCH_SIZE);
}

void Model::Destroy() {
  cudaStreamDestroy(stream);
  for (int i = 0; i < nbBindings; i++)
    cudaFree(buffers[i]);

  // destroy the engine
  context->destroy();
  engine->destroy();
}

// bool Model::InferenceFolder(const std::string &folder_name) {
//   std::vector<std::string> sample_images = readFolder(folder_name);

//   EngineInference(sample_images, outSize, buffers, bufferSize, stream);

//   // release the stream and the buffers
// }

std::vector<std::vector<float *>>
Model::EngineInference(const std::vector<cv::Mat> &image_list) {
  int index = 0;
  int batch_id = 0;
  std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
  std::vector<std::string> vec_name(BATCH_SIZE);
  std::vector<std::vector<float *>> results;
  float total_time = 0;
  for (const cv::Mat &img : image_list) {
    index++;
    if (img.data) {
      vec_Mat[batch_id] = img.clone();
      batch_id++;
    }
    if (batch_id == BATCH_SIZE or index == image_list.size()) {

      // preprocess batch images

      auto t_start_pre = std::chrono::high_resolution_clock::now();

      std::vector<float> curInput = prepareImage(vec_Mat);

      auto t_end_pre = std::chrono::high_resolution_clock::now();

      float total_pre =
          std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre)
              .count();
      std::cout << "prepare image take: " << total_pre << " ms." << std::endl;

      total_time += total_pre;
      batch_id = 0;

      if (!curInput.data()) {
        std::cout << "prepare images ERROR!" << std::endl;
        continue;
      }

      // DMA the input to the GPU,  execute the batch asynchronously, and DMA it
      // back:

      cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0],
                      cudaMemcpyHostToDevice, stream);

      // do inference

      auto t_start = std::chrono::high_resolution_clock::now();
      context->execute(BATCH_SIZE, buffers);
      auto t_end = std::chrono::high_resolution_clock::now();
      float total_inf =
          std::chrono::duration<float, std::milli>(t_end - t_start).count();
      std::cout << "Inference take: " << total_inf << " ms." << std::endl;
      total_time += total_inf;

      // Post process
      auto r_start = std::chrono::high_resolution_clock::now();
      std::vector<float *> outs;
      for (int i = 0; i < nbBindings - 1; i++) {
        float *out = new float[outSize[i] * BATCH_SIZE];
        cudaMemcpyAsync(out, buffers[i + 1], bufferSize[i + 1],
                        cudaMemcpyDeviceToHost, stream);

        outs.push_back(out);
      }
      results.push_back(outs);
      cudaStreamSynchronize(stream);

      auto r_end = std::chrono::high_resolution_clock::now();
      float total_res =
          std::chrono::duration<float, std::milli>(r_end - r_start).count();
      std::cout << "Post process take: " << total_res << " ms." << std::endl;
      total_time += total_res;
    }
  }

  std::cout << "Average processing time is " << total_time / image_list.size()
            << "ms" << std::endl;

  return results;
}

std::vector<float> Model::prepareImage(std::vector<cv::Mat> &vec_img) {
  std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT *
                            INPUT_CHANNEL);
  float *data = result.data();

  int index = 0;
  for (const cv::Mat &src_img : vec_img) {
    if (!src_img.data)
      continue;
    // cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    float ratio = float(IMAGE_WIDTH) / float(src_img.cols) <
                          float(IMAGE_HEIGHT) / float(src_img.rows)
                      ? float(IMAGE_WIDTH) / float(src_img.cols)
                      : float(IMAGE_HEIGHT) / float(src_img.rows);
    cv::Mat flt_img =
        cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);

    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3, 1.f / 255);
    
    // HWC TO CHW
    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
        cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1,
                data + channelLength * (index + 2)),
        cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1,
                data + channelLength * (index + 1)),
        cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1,
                data + channelLength * index)};
    index += 3;
    cv::split(flt_img, split_img);
  }
  std::cout<<"\n=========================================input==============================================\n\n";
  for(int i = 200 ; i<50;i++)
    std::cout<<result[i]<<" ";
  std::cout<<"\n";
  std::cout<<"\n=============================================================================================\n\n";
  return result;
}

// std::vector<std::vector<Model::FaceRes>>
// Model::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output_1,
//                    float *output_2, float *output_3, float *output_4,
//                    const int &outSize_1, const int &outSize_2,
//                    const int &outSize_3, const int &outSize_4) {

// }
