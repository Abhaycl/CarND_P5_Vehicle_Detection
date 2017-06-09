# Vehicle Detection and Tracking Project

In this project, goal is to write a software pipeline to detect vehicles in a test video and later implement on full project video from a front-facing camera on a car.

<!--more-->

[//]: # (Image References)

[image1]: /output_images/vehicle_nonvehicle_img.jpg "Sample vehicle and non-vehicle image"
[image2]: /output_images/hog_img.jpg "Sample hog image"
[image3]: /output_images/spatial-binned_img.jpg "Sample spatial binned image"
[image4]: /output_images/color-histogram_img.jpg "Sample color histogram image"
[image5]: /output_images/features_img.jpg "Sample features image"
[image6]: /output_images/windowsX-X_img.jpg "Sample slidingWarped straight image after perspective transform" 
[image7]: /output_images/heat_areas_img.jpg "Sample heat areas image" 
[image8]: /output_images/heatmap_img.jpg "Sample remove false positives image"

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
#### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second and fourth code cells of the IPython notebook.

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

I tried several combinations of parameters and the parameters that I chose were the ones that gave me the best results with the images.

I have trained a linear SVM using the following parameters HOG, spatial binned and color characteristics, with a sampling of 2000 random images.

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
n_samples = 2000
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
vehicles_features = extract_features(test_vehicles, cspace = cspace, spatial_size = spatial_size,
                                     hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell,
                                     cell_per_block = cell_per_block, hog_channel = hog_channel,
                                     spatial_feat = spatial_feat, hist_feat = hist_feat,
                                     hog_feat = hog_feat)

nonvehicles_features = extract_features(test_nonvehicles, cspace = cspace, spatial_size = spatial_size,
                                        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell,
                                        cell_per_block = cell_per_block, hog_channel = hog_channel,
                                        spatial_feat = spatial_feat, hist_feat = hist_feat,
                                        hog_feat = hog_feat)

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

#### Sliding Window Search

I decided to look for the positions in each image in a frame from 400px to 672px, there are four sizes of windows 40px, 64px, 80px and 128px that will identify the vehicles, a frame with a certain size is chosen to search vehicles where it is More likely to appear.

An overlap of 0.5 is chosen for the displacement of 0.5 towards the sides and up and down with a scale of 1.5.

```python
y_start_stop = [400, 672]
scale = 1.5
overlap = 0.5
```

```python
windows0 = slide_window(img, x_start_stop = x_start_stop, y_start_stop = y_start_stop,
                        xy_window = (40, 40), xy_overlap = (overlap, overlap))

windows1 = slide_window(img, x_start_stop = x_start_stop, y_start_stop = y_start_stop,
                        xy_window = (64, 64), xy_overlap = (overlap, overlap))

windows2 = slide_window(img, x_start_stop = x_start_stop, y_start_stop = y_start_stop,
                        xy_window = (80, 80), xy_overlap = (overlap, overlap))

windows3 = slide_window(img, x_start_stop = x_start_stop, y_start_stop = y_start_stop,
                        xy_window = (128, 128), xy_overlap = (overlap, overlap))

windows = windows0 + windows1 + windows2 + windows3
```

![Sliding window search][image6]

Following is a good example of the search of the vehicles, there is a good result in the overlap and scale of each box, applying the characteristics HOG, with a color channel YCrCb previously described.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected, the bounding boxes then overlaid.

#### Here are six frames and their corresponding heatmaps:

![Heat areas][image7]

In this part and on the heat maps previous, we can see how false positives are removed. It can be seen that for the threshold we use a value of 1 and normalize the values.

```python
heat = apply_threshold(heatmap, 1)
heatmap = np.clip(heat, 0, 255)
```

![Remove false positives][image8]


#### Discussion

---

In cases that will fail the detection of the vehicles, a better and varied selection of test images could be used, changing some parameters to obtain better results in the detection of the vehicles, possible improvement would be to use different scales for different areas, apply a better threshold for remove false positives.
