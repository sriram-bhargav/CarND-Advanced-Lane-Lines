
# Advanced Lane Finding

## Goal
Write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. Rubric points (https://review.udacity.com/#!/rubrics/571/view) are also covered in this ipython notebook.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first, second code cell of the IPython notebook.

I used chessboard calibration images (of size 9x6) provided as part of project resources to compute camera matrix and distortion coefficients.
Steps involved:

        1. Every calibrated image is converted to grayscale
        2. cv2.findChessboardCorners is used to find corners.
        3. If successful, the corners are saved in imgpoints list.
        4. As we know the size of chessboard, objpoints list is easy to construct by generating sequence ((0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)) for each image.
        
I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function.

![png](writeup_images/output_4_0.png)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

undistort() function is code cell 2 is applied on test image. Image below depicts pre and post distortion correction:

![png](writeup_images/output_6_0.png)

Note the change in white car's location :)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

undistort_and_threshold() function in code cell 6 applies color transformation and sobel operator to generated thresholded binary image.

As you can see in the images below "Saturation" (higher threshold) and "Hue" (lower threshold) channel of HSV image detects lane lines better than others. Also, absolute sobel threshold X method seem to identify lanes better than other thresholding methods.

Together with HSV color transform (H and S channel) and Sobel threshold X gradient, we obtain thresholded binary image.

Directional and Magnitude thresholds has very minimal to no effect on thresholded binary image.

![png](writeup_images/output_9_0.png)


![png](writeup_images/output_9_1.png)


![png](writeup_images/output_9_2.png)


![png](writeup_images/output_11_0.png)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

perspective_transform() in code cell 7 does perspective transformation.

When we apply a perspective transform, we need to choose four source points. As we know that camera position is fixed and road is flat most of the times, we can do fixed perspective transformation using four hard-coded source points.

We have used getPerspectiveTransform() and warpPerspective() to compute perspective transform and apply on image using following Source and Destination polygons.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 170, 720      | 320, 0        | 
| 550, 460      | 320, 720      |
| 745, 460      | 960, 0        |
| 1200, 720     | 960, 720      |


![png](writeup_images/output_14_0.png)

Perspective transformation on thresolded image:

![png](writeup_images/output_16_0.png)

Perspective transformation on test images:

![png](writeup_images/output_18_0.png)

![png](writeup_images/output_18_1.png)

![png](writeup_images/output_18_2.png)

![png](writeup_images/output_18_3.png)

![png](writeup_images/output_18_4.png)

![png](writeup_images/output_18_5.png)

![png](writeup_images/output_18_6.png)

![png](writeup_images/output_18_7.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

sliding_window_poly_fit() and polyfit_from_previous() in ipython notebook identify lane-line pixels and fit their positions with a polynomial.

Histogram peaks of bottom half of the binary thresholded image is used to find base of left and right line (check image in the cell below the code). Then I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame to identify lane-line pixels and then fit a 2 degree polynomial. 

![png](writeup_images/output_21_0.png)

![png](writeup_images/output_21_1.png)

The green shaded area in the above image shows where we searched for the lines. So, once we know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame. This is equivalent to using a customized region of interest for each frame of video, and should help us track the lanes through sharp curves and tricky conditions. If we lose track of the lines, we go back to your sliding windows search or other method to rediscover them.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

curvature() function in ipython notebook is responsible for computing radius of curvature and vehicle offset with center.

I used function defined in https://www.intmath.com/applications-differentiation/8-radius-curvature.php for radius of curvature.

Vehicle position estimate depends on the road curving left or right. Polynomial evaluation (of lane lines) happens at the top of the images, instead of the bottom of the screen (where the car is). So, we used value close to 720 for calculating polynomial intercept.

`
 car_pos = width / 2
 lane_center = (left_fitx[719] + right_fitx[719]) / 2
 vehicle_offset = (lane_center-car_pos)*xm_per_pix
`
left_fitx, right_fitx are polyfits of left and right lanes respectively. xm_per_pix is meters per pixel in x dimension.


### Rubric point 7

#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

process_image() in the notebook takes care of detecting lane lines and displaying on top of the image.

![png](writeup_images/output_29_1.png)


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

https://youtu.be/mv0o0mV_ER8


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Most time consuming part of the pipeline is thresholding an image. Figuring out right combination of channels (in image) and sobel operators to discard unnecessary pixels which are not part of lane lines was challenging. 

As you can see in 'challenge_output.mp4' (generated from challenge_video.mp4), my pipeline failed to identify lane lines because the lane is damaged slightly (with a line in the middle). My implementation assumes that there will be only two lines in the warped image. In the challenge video, thresolded warped image will have three 3 lines. Histogram peak method i used to find base of left and right lines will not work because of new line in the middle.

There are many ways to solve this issue. I will mention one simple method that can be used to solve this issue: 
1. Find the base of the line in the middle and traverse up to find the lane and discard all the pixels near that polyfit lane
2. Run the orginal histogram peak method to identify lanes.
