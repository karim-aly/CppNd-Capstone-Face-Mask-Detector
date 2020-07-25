#include <iostream>

#include "FaceMaskDetector.h"


FaceMaskDetector::FaceMaskDetector(std::string &faceMaskModelPath, std::string &faceMaskCkptPath, std::string &faceDetectorModelPath, const float faceDetectorConfidence) : faceMaskModel(faceMaskModelPath), faceMaskInputNode(faceMaskModel, "face_model/input_node"), faceMaskOutputNode(faceMaskModel, "face_model/output_node") {
    // Set Face Detector Model Confidence Threshold
    this->faceDetectorConfidence = faceDetectorConfidence;

    std::cout << "[INFO] Loaded TensorFlow Face Mask Detector Model\n";

    // Load Caffe Res10 Face Detector Model
    std::string prototxtPath = faceDetectorModelPath + "/deploy.prototxt";
    std::string weightsPath = faceDetectorModelPath + "/res10_300x300_ssd_iter_140000.caffemodel";
    this->faceModel = cv::dnn::readNet(prototxtPath, weightsPath);

    std::cout << "[INFO] Loaded Caffe Face Detector Model\n";

    // Load Face Mask Detector Model
    //this->faceMaskModel = std::move(Model(faceMaskModelPath));

    // Restore The Weights Saved in The Checkpoint Files
    // this->faceMaskModel.init();
    // this->faceMaskModel.restore(faceMaskCkptPath);

    // Find The Input Node in The Model Graph
    //this->faceMaskInputNode  = std::move(Tensor(this->faceMaskModel, "face_model/input_node"));

    // Find The Output Node in The Model Graph
    //this->faceMaskOutputNode = std::move(Tensor(this->faceMaskModel, "face_model/output_node/Softmax"));
}


FaceMaskDetector::~FaceMaskDetector() {
    // TODO
}

template <typename T>
T* assign(T* array, std::vector<T> &&v) {
    for (unsigned int i=0; i<v.size(); i++) {
        array[i] = v[i];
    }
    return array;
}

float fpixel(cv::Mat &m, std::vector<int> &&v) {
    return m.at<cv::Vec3f>(v[0], v[1])[v[2]];//(&v[0])[*v.end()];
}

uint8_t bpixel(cv::Mat &m, std::vector<int> &&v) {
    return m.at<cv::Vec3b>(v[0], v[1])[v[2]];
}

std::pair<std::vector<cv::Mat>, std::vector<cv::Rect>> FaceMaskDetector::detectFaces(cv::Mat &frame) {
    // grab the dimensions of the frame
    int h = frame.rows;
    int w = frame.cols;

    std::cout << "[INFO] detectFaces: frame.h=" << h << ", frame.w=" << w << ", channels=" << frame.channels() << '\n';

    // construct a blob from the frame
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));

    std::cout << "[INFO] detectFaces: created blob with blob.h=" << blob.rows << ", blob.w=" << blob.cols << '\n';

    // pass the blob through the network
    this->faceModel.setInput(std::move(blob));

    std::cout << "[INFO] detectFaces: passed frame blob as input to face detector model\n";

    // obtain the face detections
    cv::Mat detections = this->faceModel.forward();

    std::cout << "[INFO] detectFaces: got the detections from the faces model, detections.count=" << detections.size[2] << '\n';

    // create empty vectors to hold the results
    std::vector<cv::Mat> faces;
    std::vector<cv::Rect> faceLocations;

    // loop over the detections
    double confidence;
    int startX, startY, endX, endY;
    int idx[4];
    for (int i = 0; i < detections.size[2]; i++) {
        confidence = detections.at<float>(assign(idx, {0,0,i,2}));

        // filter out weak detections by ensuring the confidence is
        // greater than the minimum confidence
        if (confidence > this->faceDetectorConfidence) {
            std::cout << "[INFO] detectFaces: detection[" << i << "].confidence=" << confidence << '\n';

            // compute the (x, y)-coordinates of the bounding box for the object
            // ensure the bounding boxes fall within the dimensions of the frame
            startX = std::max(static_cast<int>(detections.at<float>(assign(idx, {0,0,i,3})) * w), 0);
            startY = std::max(static_cast<int>(detections.at<float>(assign(idx, {0,0,i,4})) * h), 0);
            endX   = std::min(static_cast<int>(detections.at<float>(assign(idx, {0,0,i,5})) * w), w-1);
            endY   = std::min(static_cast<int>(detections.at<float>(assign(idx, {0,0,i,6})) * h), h-1);

            std::cout << "[INFO] detectFaces: detection[" << i << "].startX=" << startX << '\n';
            std::cout << "[INFO] detectFaces: detection[" << i << "].startY=" << startY << '\n';
            std::cout << "[INFO] detectFaces: detection[" << i << "].endX=" << endX << '\n';
            std::cout << "[INFO] detectFaces: detection[" << i << "].endY=" << endY << '\n';

            // extract the face ROI
            cv::Rect rect(startX, startY, endX-startX, endY-startY);
            cv::Mat roi = frame(rect).clone();

            // cv::namedWindow("ROI Image", cv::WINDOW_AUTOSIZE);
            // cv::imshow("ROI Image", roi);
            // cv::waitKey(1000);

            // convert from BGR to RGB channel
#if (CV_VERSION_MAJOR >= 4)
            cv::cvtColor(roi, roi, cv::COLOR_BGR2RGB);
#else
            cv::cvtColor(roi, roi, CV_BGR2RGB);
#endif
            std::cout << "[INFO] detectFaces: detection[" << i << "] -> extracted ROI and converted color channels to RGB\n";
            std::cout << "[INFO] detectFaces: detection[" << i << "] -> data= " << +bpixel(roi,{0,0,0}) << " " << +bpixel(roi,{10,10,1}) << " " << +bpixel(roi,{40,40,2}) << " " << +bpixel(roi,{140,140,0}) << '\n';

            // resize image to match face mask model input size
            cv::resize(roi, roi, cv::Size(224, 224));

            std::cout << "[INFO] detectFaces: detection[" << i << "] -> resize image to match face mask model input size \n";
            std::cout << "[INFO] detectFaces: detection[" << i << "] -> roi.h=" << roi.rows << ", roi.w=" << roi.cols << ", roi.channels=" << roi.channels() << ", roi.dims=" << roi.dims << '\n';

            std::cout << "[INFO] detectFaces: detection[" << i << "] -> data=" << +bpixel(roi,{0,0,0}) << " " << +bpixel(roi,{10,10,1}) << " " << +bpixel(roi,{40,40,2}) << " " << +bpixel(roi,{140,140,0}) << '\n';
            // std::cout << "roi = " << '\n' << " "  << roi(cv::Rect(0,0,10,10)) << '\n' << '\n';

            // normalize the image values to be between -1 and 1
            cv::normalize(roi, roi, -1.0, 1.0, cv::NORM_MINMAX, CV_32FC3);

            std::cout << "[INFO] detectFaces: detection[" << i << "] -> normalized data values to match face mask model input range [-1.0, 1.0] \n";
            std::cout << "[INFO] detectFaces: detection[" << i << "] -> data=" << +fpixel(roi,{0,0,0}) << " " << +fpixel(roi,{10,10,1}) << " " << +fpixel(roi,{40,40,2}) << " " << +fpixel(roi,{140,140,0}) << '\n';

            faces.emplace_back(std::move(roi));
            faceLocations.emplace_back(std::move(rect));
        }
    }

    return {faces, faceLocations};
}

std::vector<std::pair<bool, float>> FaceMaskDetector::detectMasks(std::vector<cv::Mat> &&faces) {
    std::vector<float> img_data;
    std::vector<float> predections;
    std::vector<std::pair<bool, float>>  faceMaskDetected;

    // loop over all faces detected
    for (cv::Mat &input : faces) {
        std::cout << "[INFO] detectMasks: face[] -> data=" << +fpixel(input,{0,0,0}) << " " << +fpixel(input,{10,10,1}) << " " << +fpixel(input,{40,40,2}) << " " << +fpixel(input,{140,140,0}) << '\n';

        // Feed input image pixels data in The Input Tensor
        img_data.assign(input.data, input.data + input.total() * input.channels());
        this->faceMaskInputNode.set_data(img_data, {1, 224, 224, 3});

        std::cout << "[INFO] detectMasks: image roi passed to the input tensor of face mask model\n";

        // Run inference on The Model with the image data tensor
        this->faceMaskModel.run(this->faceMaskInputNode, this->faceMaskOutputNode);

        std::cout << "[INFO] detectMasks: ran inference on the model using the image roi\n";

        // Get the predections result
        predections = std::move(this->faceMaskOutputNode.get_data<float>());
        std::cout << "[INFO] detectMasks: mask = " << predections[0]*100 << ", no_mask = " << predections[1]*100 << " \n";

        faceMaskDetected.push_back({predections[0] > predections[1], std::max(predections[0], predections[1])});
    }

    return faceMaskDetected;
}

std::pair<std::vector<cv::Rect>, std::vector<std::pair<bool, float>>> FaceMaskDetector::detectAndPredictMask(cv::Mat &frame) {
    // Run Faces Detection Model and Get The Faces as cv::Mat Objects and Their Locations as cv::Rect Objects
    std::pair<std::vector<cv::Mat>, std::vector<cv::Rect>> faces_detector_result = std::move(detectFaces(frame));

    // Get The Faces Vector from the Pair Object
    std::vector<cv::Mat> faces = std::move(faces_detector_result.first);
    std::vector<cv::Rect> facesLocations = std::move(faces_detector_result.second);

    // Check if no faces was found and if that's case return immeditely
    if (faces.size() == 0) {
        return {};
    }

    // Run Face Mask Detection Model on The Faces Found
    std::vector<std::pair<bool, float>> mask_detector_result = std::move(detectMasks(std::move(faces)));

    return {facesLocations, mask_detector_result};
}