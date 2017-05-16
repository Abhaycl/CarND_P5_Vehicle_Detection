# Vehicle Detection and Tracking Project

In this project, goal is to write a software pipeline to detect vehicles in a test video and later implement on full project video from a front-facing camera on a car.

<!--more-->

[//]: # (Image References)

#### How to run the program

```sh
jupyter notebook main.ipynb
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
| main.ipynb        | Main python file that runs the program with all the helper functions and the main pipeline  |
| test_video.mp4    | Video test                                                                                  |
| project_video.mp4 | Video of the complete project                                                               |
|                   |                                                                                             |
| test_images       | Folder with road images used to test the pipeline                                           |
| output_videos     | Folder to store output videos                                                               |
| vehicles          | Folder with images of examples of vehicles                                                  |
| non-vehicles      | Folder with images of examples of non-vehicles                                              |
|                   |                                                                                             |


---

Most of the output images are in the file main.ipynb where the results of each of the processes are shown, as well as test samples that I have made by changing some parameters or with different test images, the generated videos are in the output_videos folder. In the processes is commented its functionality


#### Discussion

---


In cases that will fail the detection of the vehicles, a better and varied selection of test images could be used, changing some parameters to obtain better results in the detection of the vehicles