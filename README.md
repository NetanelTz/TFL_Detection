# TFL_Detection
# Mobileye-Traffic-Lights-Detection
This project detect traffic lights in images, find thier positions and calculate distance for each traffic light to the camera.
The project consists of 4 parts, as described below.

## Detect
First part: detect TFLs in images by unique kernel and mark the source light. 

(Source code file - RGB_filtering.py).

![Image](./Figure_1.jpg)

## Predict
Second part: training a model to recognize TFLs.

(Source code directory - dataset_training).

![Image](./TFL-Predict.png)

## Detect & Calculate Distance
Third part: calculate distance to tfl by geometric and linear algebra calculations.

(Source code directory - SFM).

![Image](./Figure_2.png)

## Integration
Last part: integrate all parts to one functional software.

(Source code files - main.py, controller.py).

## Collaborators
- [Omer Hadad](https://github.com/omerhad)
- [Matan Omesi](https://github.com/matan1346)
- [Ariel Haser](https://github.com/arielhaser)
