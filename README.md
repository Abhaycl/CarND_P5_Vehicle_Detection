# Vehicle Detection and Tracking Project

In this project, goal is to write a software pipeline to detect vehicles in a test video and later implement on full project video from a front-facing camera on a car.

<!--more-->

[//]: # (Image References)

#### How to run the program

```sh
jupyter notebook main.ipynb

Once the project has been opened in the cell menu, choose Run All
```

**The process we will fallow can be summarized as:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Train a classifier Linear SVM classifier.
* Color transform and binned color features, as well as histograms of color to HOG feature vector.
* Normalize features and randomize a selection for training and testing.
* Implementation a sliding-window technique and use trained classifier to search for vehicles in images.
* A heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


The summary of the files and folders int repo is provided in the table below:

| File/Folder       | Definition                                                                                  |
| :---------------- | :------------------------------------------------------------------------------------------ |
| P5.ipynb          | Main python file that runs the program with all the helper functions and the main process.  |
| calibrations.p    | File containing the calibration parameters for the detection of the lines obtained in the   |
|                   | practice 4 of the course.                                                                   |
| Lines.py          | File that includes the class to store tracking information and the windows search function. |
| test_video.mp4    | Video test.                                                                                 |
| project_video.mp4 | Video of the complete project.                                                              |
|                   |                                                                                             |
| test_images       | Folder with road images used to test the pipeline.                                          |
| output_videos     | Folder to store output videos.                                                              |
| vehicles          | Folder with images of examples of vehicles.                                                 |
| non-vehicles      | Folder with images of examples of non-vehicles.                                             |
|                   |                                                                                             |


---

Most of the output images are in the P5.ipynb file where the results of each of the processes are displayed, the generated videos are in the output_videos folder. In the lines of code it's commented some of the functionalities, also includes the processing and the video of the challenge in the detection of the lines and the vehicles at the same time.

Note: Due to space problems for vehicle and non-vehicle folders, it is only necessary to include the images with which I train the classifier, which correspond to the training dataset provided for this project (vehicle and non-vehicle images) are in the .png format.


#### Discussion

---


In cases that will fail the detection of the vehicles, a better and varied selection of test images could be used, changing some parameters to obtain better results in the detection of the vehicles
