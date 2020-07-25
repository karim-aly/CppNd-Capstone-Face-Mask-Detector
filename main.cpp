//
// Created by Karim Aly on 24/07/20.
//

#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "FaceMaskDetector.h"
#include "BoundingBoxesWindow.h"


int main() {
    std::string faceMaskModelPath = "./../face_mask_detector_model/face_mask_detector/saver_data_v2/frozen_optimized_model.pb";
    std::string faceMaskCheckpointPath = "./../face_mask_detector_model/face_mask_detector/saver_data_v2/train.ckpt";
    std::string faceModelPath = "./../face_detector";

    std::string exampleImgPath = "./../examples/example_02.jpg";

    // Load Face Mask Detector Model
    FaceMaskDetector model(faceMaskModelPath, faceMaskCheckpointPath, faceModelPath);

    // Read image
    cv::Mat img = cv::imread(exampleImgPath, cv::IMREAD_COLOR);

    // Detect Faces and Get Predictions
    std::pair<std::vector<cv::Rect>, std::vector<std::pair<bool, float>>> predictions = model.detectAndPredictMask(img);

    // Draw the results on the frame as Bounding Boxes with Labels of the result
    BoundingBoxesWindow window("frame");
    window.drawBoundingBoxes(img, std::move(predictions));

    // Wait until 'q' key is pressed
    while(true) {
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        }
    }

    return 0;
}
