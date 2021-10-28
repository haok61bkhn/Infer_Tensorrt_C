#include "Model.h"
#include <iostream>
#include<fstream> 
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  // cv::Mat img1 = cv::imread("../samples/test1.jpg");
  // cv::Mat img2 = cv::imread("../samples/test2.jpg");
  // cv::Mat img3 = cv::imread("../samples/test3.jpg");
  cv::Mat img = cv::imread("../127.jpg");
  std::vector<cv::Mat> list_imgs = {img};

  Model CenterFace("../config.yaml");
  CenterFace.LoadEngine();

  std::vector<std::vector<float *>> res_infers =
      CenterFace.EngineInference(list_imgs);

  //print output 
  

  std::cout<<"\n-------------------------OUTPUT------------------------------\n";
  for (int i = 0; i < list_imgs.size();i++) {
    std::vector<float *> res_infer = res_infers[i];
    for (int j = 0; j < CenterFace.nbBindings - 1; j++) {
      float* out = res_infer[j];
      std::cout << "output "<< j << " with outsize "<< CenterFace.outSize[j]<< " : \n";
      // for (int k = 0; k < CenterFace.outSize[j]; k++)
      std::ofstream outfile (std::to_string(j)+"OutputFile2.dat");
      for (int k = 0; k <CenterFace.outSize[j]; k++)
        outfile << std::setw(4) << out[k] <<std::endl;
      std::cout << "\n\n";
      outfile.close();
    }
  }
  CenterFace.Destroy();

  return 0;
}