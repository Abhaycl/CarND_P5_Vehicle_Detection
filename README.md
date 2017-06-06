# Vehicle Detection and Tracking Project

In this project, goal is to write a software pipeline to detect vehicles in a test video and later implement on full project video from a front-facing camera on a car.

<!--more-->

[//]: # (Image References)

[image1]: /output_images/vehicle_nonvehicle_img.jpg "Sample vehicle and non-vehicle image"
[image2]: /output_images/hog_img.jpg "Sample hog image"
[image3]: /output_images/spatial-binned_img.jpg "Sample spatial binned image"
[image4]: /output_images/color-histogram_img.jpg "Sample color histogram image"
[image5]: /output_images/features_img.jpg "Sample features image"
[image6]: /output_images/straight/straight_lines1_compare.png "Warped straight image after perspective transform" 
[image7]: /output_images/warped/test3_compare.png "Warped test image after perspective transform" 
[image8]: /output_images/windows/test5_compare.png "Windows around centroids on warped image"
[image9]: /output_images/lanelines/test2_compare.png "Image with detected lane lines"
[image10]: /output_images/full/test4_compare.png "Final image with lane lines, car offset and road curvature"

#### How to run the program

```sh
jupyter notebook main.ipynb

Once the project has been opened in the cell menu, choose Run All
```

Note: Due to space problems for vehicle and non-vehicle folders, it is only necessary to include the images with which I train the classifier, which correspond to the training dataset provided for this project (vehicle and non-vehicle images) are in the .png format.

**The process we will fallow can be summarized as:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Color transform and binned color features, as well as histograms of color to HOG feature vector.
* Normalize features and randomize a selection for training and testing.
* Train a classifier Linear SVM classifier.
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
| output_images     | Folder to store output images.                                                              |
| output_videos     | Folder to store output videos.                                                              |
| vehicles          | Folder with images of examples of vehicles.                                                 |
| non-vehicles      | Folder with images of examples of non-vehicles.                                             |
|                   |                                                                                             |


---
###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second and fourth code cells of the IPython notebook

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    if vis == True:
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = False, 
                                  visualise = vis, feature_vector = feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                       cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = False,
                       visualise = vis, feature_vector = feature_vec)
        return features
```

I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:

![vehicle and non-vehicle image][image1]

I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.

Here is an example using the YCrCb color space and HOG parameters of orientations = 9, pixels_per_cell = (8, 8) and cells_per_block = (2, 2):

![Hog image][image2]

I tried several combinations of parameters and the parameters that I chose were the ones that gave me the best results with the images

I have trained a linear SVM using the following parameters HOG, spatial binned and color characteristics

```python
orient = 9
pix_per_cell = 8
cell_per_block = 2
cspace = 'YCrCb'
hog_channel = ALL
spatial_size = (64, 64)
hist_bins = 64
spatial_feat = True
hist_feat = True
hog_feat = True
```

```python
vehicle_binning = bin_spatial(vehicle_image, spatial_size)
nonvehicle_binning = bin_spatial(nonvehicle_image, spatial_size)
```

![Spatial binned image][image3]

```python
vehicle_color_hist = color_hist(vehicle_image, hist_bins)
nonvehicle_color_hist = color_hist(nonvehicle_image, hist_bins)
```

![Color histogram image][image4]

```python
vehicles_features = extract_features(test_vehicles, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins,
                                     orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,
                                     hog_channel = hog_channel, spatial_feat = spatial_feat, hist_feat = hist_feat,
                                     hog_feat = hog_feat)

nonvehicles_features = extract_features(test_nonvehicles, cspace = cspace, spatial_size = spatial_size,
                                        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell,
                                        cell_per_block = cell_per_block, hog_channel = hog_channel,
                                        spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)

print(round(time.time() - t, 2), 'Seconds to compute the features.\\n')
x = np.vstack((vehicles_features, nonvehicles_features)).astype(np.float64)
# Fit a per-column scaler
x_scaler = StandardScaler().fit(x)
# Apply a per-column scaler
scaled_x = x_scaler.transform(x)
# Define the labels vector
y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(nonvehicles_features))))
```

![Features image][image5]

Most of the output images are in the P5.ipynb file where the results of each of the processes are displayed, the generated videos are in the output_videos folder. In the lines of code it's commented some of the functionalities, also includes the processing and the video of the challenge in the detection of the lines and the vehicles at the same time.

Note: Due to space problems for vehicle and non-vehicle folders, it is only necessary to include the images with which I train the classifier, which correspond to the training dataset provided for this project (vehicle and non-vehicle images) are in the .png format.


#### Discussion

---


In cases that will fail the detection of the vehicles, a better and varied selection of test images could be used, changing some parameters to obtain better results in the detection of the vehicles, possible improvement would be to use different scales for different areas
