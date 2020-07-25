#ifndef BOUNDING_BOXES_WINDOW_H
#define BOUNDING_BOXES_WINDOW_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class BoundingBoxesWindow {

  public:
    BoundingBoxesWindow(const std::string &windowName);
    ~BoundingBoxesWindow();
    BoundingBoxesWindow(const BoundingBoxesWindow &source) = delete;
    BoundingBoxesWindow(BoundingBoxesWindow &&source) = default;
    BoundingBoxesWindow& operator=(const BoundingBoxesWindow &source) = delete;
    BoundingBoxesWindow& operator=(BoundingBoxesWindow &&source) = default;

    cv::Mat& drawBoundingBoxes(cv::Mat &frame, std::pair<std::vector<cv::Rect>, std::vector<std::pair<bool, float>>> predictions);

  private:
    std::string windowName;
};



#endif /* BOUNDING_BOXES_WINDOW_H */