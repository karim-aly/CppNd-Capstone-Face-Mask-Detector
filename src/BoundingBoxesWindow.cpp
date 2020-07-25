#include <iostream>

#include "BoundingBoxesWindow.h"


BoundingBoxesWindow::BoundingBoxesWindow(const std::string &windowName) {
    this-> windowName = windowName;
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
}

BoundingBoxesWindow::~BoundingBoxesWindow() {
    cv::destroyWindow(this-> windowName);
}

cv::Mat& BoundingBoxesWindow::drawBoundingBoxes(cv::Mat &frame, std::pair<std::vector<cv::Rect>, std::vector<std::pair<bool, float>>> predictions) {
    std::vector<cv::Rect> facesLocations = std::move(predictions.first);
    std::vector<std::pair<bool, float>> mask_detector_result = std::move(predictions.second);

    std::string label;
    bool predictedMask;
    float precision;

    cv::Rect location;

    // loop over all predictions found
    for(int i=0; i<facesLocations.size(); i++) {

        predictedMask = mask_detector_result[i].first;
        precision = mask_detector_result[i].second;

        location = facesLocations[i];

        std::cout << facesLocations[i] << '\n';
        std::cout << predictedMask << " " << precision << '\n';

        // determine the class label
        label = (predictedMask ? "Mask " : "No Mask ") + std::to_string(precision*100);

        // determine color we'll use to draw the bounding box and text
        cv::Scalar color = predictedMask ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

        // add label text the output frame
        cv::Point point = cv::Point(location.x, location.y-10);
        cv::putText(frame, label, point, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 2);

        // add the bounding box rectangle on the output frame
        cv::rectangle(frame, location, color, 2);
    }

    cv::imshow(this->windowName, frame);

    return frame;
}