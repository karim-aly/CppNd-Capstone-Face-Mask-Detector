#ifndef FACE_MASK_DETECTOR_H
#define FACE_MASK_DETECTOR_H

#include "Model.h"
#include "Tensor.h"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class FaceMaskDetector {

  public:
    FaceMaskDetector(std::string &faceMaskModelPath, std::string &faceMaskCkptPath, std::string &faceDetectorModelPath, const float faceDetectorConfidence=0.5f);
    ~FaceMaskDetector();
    FaceMaskDetector(const FaceMaskDetector &source) = delete;
    FaceMaskDetector(FaceMaskDetector &&source) = default;
    FaceMaskDetector& operator=(const FaceMaskDetector &source) = delete;
    FaceMaskDetector& operator=(FaceMaskDetector &&source) = default;

    std::pair<std::vector<cv::Rect>, std::vector<std::pair<bool, float>>> detectAndPredictMask(cv::Mat &frame);

  private:
    std::pair<std::vector<cv::Mat>, std::vector<cv::Rect>> detectFaces(cv::Mat &frame);
    std::vector<std::pair<bool, float>> detectMasks(std::vector<cv::Mat> &&faces);

    Model faceMaskModel;
    Tensor faceMaskInputNode;
    Tensor faceMaskOutputNode;

    cv::dnn::Net faceModel;
    float faceDetectorConfidence;
};



#endif /* FACE_MASK_DETECTOR_H */