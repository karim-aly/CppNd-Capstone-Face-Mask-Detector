# CppNd-Capstone-Face-Mask-Detector
Face Mask Detector using OpenCV and TensorFlow in C++

# Dataset
Dataset used is the one from this repository [prajnasb/observations](https://github.com/prajnasb/observations)

The Project was developed and tested on Ubuntu 18.04

## Requirements:
- OpenCV (V3.2+) [Setup Guide](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)
- TensorFlow (V1.15.0) [TensorFlow C++ API Install Guide](https://www.tensorflow.org/install/lang_c)

## Build Instructions
Stating from Root Directory, Do the following instructions:
```bash
  mkdir build && cd build
  cmake ..
  make
```
## Running
`./face_mask_detect`

## Project Structure
The project is using an Object Oriented Programming Structure.
- The `main.cpp` file is the entry point to the program.

- The `./include/Model.h`, `./src/Model.cpp`, `./include/Tensor.h` and `./src/Tensor.cpp` files contains the definitions and declarations for The 2 Classes `Tensor` and `Model` which are used to wrap TensorFlow APIs which load a saved TensorFlow Model and Access Tensor Nodes (Input and Output Nodes specifically).

- The `./include/FaceMaskDetector.h` and `./src/FaceMaskDetector.cpp` files contains the definition and declaration for The Class `FaceMaskDetector` which is used to wrap and load the TensorFlow Mask Detector Model along with The Caffe Faces Detector Model. This class takes care of all prediction and inference work.

- The `./include/BoundingBoxesWindow.h` and `./src/BoundingBoxesWindow.cpp` files contains the definition and declaration for The Class `BoundingBoxesWindow` which is used to wrap and manage an OpenCV Window for Showing the Result and Takes as an input the predictions results from the Class `FaceMaskDetector` and Draw Bounding Boxes around Faces and Label each box as a Face with Mask or not.

- The Diretory `./face_detector` contains the Caffe Faces Detector Model and Weights.

- The Diretory `./face_mask_detector_model` contains the TensorFlow Face Mask Detector Model and Weights along with the Python Files and Notebooks used for the training phase, Saving the model in `ProtoBuf` and `HDF5` Formats ,and Python Scripts used for Testing the model accuracy.

- The Diretory `./examples` contains test images used to test the c++ program.

## Behaviour
Once the project is built, The project can be ran directly and The Result is a Window containing the Example Image Used in `main.cpp` along with Bounding Boxes around the faces and each has a label of the predictions of The people faces detected wearing face masks or not.

The window can be closed by pressing on the `q` button.

This behaviour can be extended in the future to run on a Webcam Stream and Label in Realtime the Face Mask Detections.


## Rubrics Coverage
- `The project code must compile and run without errors. ` <br>
  > Project Builds and Runs without any warnings using cmake and make.

- `A variety of control structures are used in the project.`<br>
  > Usage of `vector` and `pair` objects, e.g: ./main.cpp#31

- `The project code is clearly organized into functions.`<br>
  > Project Code is organized into functions and classes

- `The project reads data from an external file or writes data to a file as part of the necessary operation of the program.`<br>
  > Project reads images and model files from the disk storage, e.g: ./main.cpp#28

- `The project accepts input from a user as part of the necessary operation of the program.`<br>
  > Program waits for the user to press the `q` key to close the window, e.g: ./main.cpp#39

- `The project code is organized into classes with class attributes to hold the data, and class methods to perform tasks.`<br>
  > As described in the project structure, The project uses mainly classes.

- `All class data members are explicitly specified as public, protected, or private.`<br>
 `All class member functions document their effects, either through function names, comments, or formal documentation. Member functions do not change program state in undocumented ways.` <br>
 `Appropriate data and functions are grouped into classes. Member data that is subject to an invariant is hidden from the user. State is accessed via member functions.` <br>
  > This can be found in any class defined, e.g: `./include/FaceMaskDetector.h`

- `One function is declared with a template that allows it to accept a generic parameter`<br>
  > Helper Function `assign` in ./src/FaceMaskDetector.cpp#39 

- `At least two variables are defined as references, or two functions use pass-by-reference in the project code.`<br>
  > FaceMaskDetector class constuctor ./src/FaceMaskDetector.cpp#6 
  
- `At least two variables are defined as references, or two functions use pass-by-reference in the project code.`<br>
  > FaceMaskDetector class constuctor ./src/FaceMaskDetector.cpp#6 

- `The project follows the Resource Acquisition Is Initialization pattern where appropriate, by allocating objects at compile-time, initializing objects when they are declared, and utilizing scope to ensure their automatic destruction.` <br>
  > This is followed in the whole project.
 
- `For all classes, if any one of the copy constructor, copy assignment operator, move constructor, move assignment operator, and destructor are defined, then all of these functions are defined.`<br>
  > This is followed in all classes defined.
  
- `For classes with move constructors, the project returns objects of that class by value, and relies on the move constructor, instead of copying the object.`<br>
  > This is followed in all classes defined.
