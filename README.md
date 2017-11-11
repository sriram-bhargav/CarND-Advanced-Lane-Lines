
# Advanced Lane Finding (writeup + code)

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


### Rubric point 3

#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

As you can see below "Saturation" (higher threshold) and "Hue" (lower threshold) channel of HSV image detects lane lines better than others. Also, absolute sobel threshold X method seem to identify lanes better than other thresholding methods.

Together with HSV color transform (H and S channel) and Sobel threshold X gradient, we obtain thresholded binary image.

Directional and Magnitude thresholds has very minimal to no effect on thresholded binary image.

![png](writeup_images/output_9_0.png)



![png](writeup_images/output_9_1.png)



![png](writeup_images/output_9_2.png)


![png](writeup_images/output_11_0.png)


### Perspective Transform
### Rubric point 4

#### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

When we apply a perspective transform, we need to choose four source points. As we know that camera position is fixed and road is flat most of the times, we can do fixed perspective transformation using four hard-coded source points.

We have used getPerspectiveTransform and warpPerspective to compute perspective transform and apply on image using following src and dst polygons.

|  src (X, Y) |        
|------|------|       
|  170 | 720  |        
|  550 | 460  |        
|  745 | 460  |        
| 1200 | 720  |        


|  dst (X, Y) |
|------|------| 
|  320 |   0  |
|  320 | 720  |
|  960 |   0  |
|  960 | 720  |


```python
def perspective_transform(img):
    h,w = img.shape[:2]
    # Define 4 source points (which takes shape of trapezoid)
    src = np.float32([[170, img.shape[0]], [550, 460], 
                      [745, 460], [1200, img.shape[0]]])
    # Define 4 destination points (which takes shape of rectangle)
    dst = np.float32([[100, img.shape[0]], [100, 0], 
                      [1100, 0], [1100, img.shape[0]]])
    # Compute the perspective transform, M, given source and destination points.
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform.
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp an image using the perspective transform, M
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv, src

def perspective_transform_helper(img):
    warped,_,_,_ = perspective_transform(img)
    return warped
```


```python
def display_unwarped(img, cmap='hsv'):
    result,_,_,src = perspective_transform(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    f.tight_layout()

    ax1.imshow(img, cmap=cmap)
    ax1.set_title('Original Image', fontsize=15)
    x = [src[0][0],src[1][0],src[2][0],src[3][0],src[0][0]]
    y = [src[0][1],src[1][1],src[2][1],src[3][1],src[0][1]]
    ax1.plot(x, y, color='#f75b44', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)

    ax2.imshow(result, cmap=cmap)
    ax2.set_title('Unwarped Image', fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

img = mpimg.imread('test_images/test4.jpg')
display_unwarped(img)
```


![png](writeup_images/output_14_0.png)


Let's apply perspective transformation on thresolded image.


```python
img = mpimg.imread('test_images/test4.jpg')
display_unwarped(undistort_and_threshold(img), cmap='gray')
```


![png](writeup_images/output_16_0.png)


Now let's see how transformation looks like on test images


```python
images = glob.glob('test_images/*.jpg')
for fname in images:
    img = mpimg.imread(fname)
    display_unwarped(undistort_and_threshold(img), cmap='gray')
```


![png](writeup_images/output_18_0.png)



![png](writeup_images/output_18_1.png)



![png](writeup_images/output_18_2.png)



![png](writeup_images/output_18_3.png)



![png](writeup_images/output_18_4.png)



![png](writeup_images/output_18_5.png)



![png](writeup_images/output_18_6.png)



![png](writeup_images/output_18_7.png)


### Finding the lanes

### Rubric point 5
#### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Histogram peaks of bottom half of the image is used to find base of left and right line (check image in the cell below the code). Then I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame to identify lane-line pixels and then fit a 2 degree polynomial. 


```python
# Returns polyfit for left and right lanes. 
def sliding_window_poly_fit(binary_warped, visualize=True):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if visualize:
        # Print histogram from sliding window polyfit for example image
        plt.plot(histogram)
        plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #if visualize:
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 3) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    if left_fit is not None and right_fit is not None:
        # Generate x and y values for plotting
        ploty, left_fitx, right_fitx = generate_xy_for_plotting(binary_warped.shape[0], left_fit, right_fit)

        if visualize:
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                          ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                          ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)

    return left_fit, right_fit

def generate_xy_for_plotting(height, left_fit, right_fit):
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx
```


```python
img = mpimg.imread('test_images/test4.jpg')
binary_warped = perspective_transform_helper(undistort_and_threshold(img))
left_fit, right_fit = sliding_window_poly_fit(binary_warped)
```


![png](writeup_images/output_21_0.png)



![png](writeup_images/output_21_1.png)


The green shaded area in the above image shows where we searched for the lines. So, once we know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame. This is equivalent to using a customized region of interest for each frame of video, and should help us track the lanes through sharp curves and tricky conditions. If we lose track of the lines, we go back to your sliding windows search or other method to rediscover them.


```python
def polyfit_from_previous(binary_warped, left_fit_prev, right_fit_prev):
    left_fit, right_fit = (None, None)
    if left_fit_prev is None or right_fit_prev is None:
        return left_fit, right_fit
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + 
    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + 
    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + 
    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + 
    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit
```

### Finding Radius of Curvature

### Rubric point 6

#### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

    1. I used https://www.intmath.com/applications-differentiation/8-radius-curvature.php for estimating radius of curvature.
    2. Difference in car's position (center of the image) and center of left, right lane lines is used for estimating car's offset from center. 


```python
def curvature(width, ploty, left_fitx, right_fitx):
    # Compute curve radius    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the radii of curvature
    left_curverad =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

    # Compute car's offset from center
    car_pos = width / 2
    lane_center = (left_fitx[0] + right_fitx[0]) / 2

    vehicle_offset = (lane_center-car_pos)*xm_per_pix
    
    # Now our radius of curvature is in meters
    radius_of_curvature = np.mean([left_curverad, right_curverad])
    return radius_of_curvature, vehicle_offset
```

### Rubric point 7

#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


```python
def draw_lane_area(undist, warped, Minv, left_fit, right_fit, show_curve_data=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty, left_fitx, right_fitx = generate_xy_for_plotting(warped.shape[0], left_fit, right_fit)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if show_curve_data:
        font = cv2.FONT_HERSHEY_COMPLEX
        radius_of_curvature, vehicle_offset = curvature(warped.shape[1], ploty, left_fitx, right_fitx)
        cv2.putText(result, "Lane Curve Radius(m): {0:0.2f}".format(radius_of_curvature),
                    (20,70), font, 2,(255,255,0),2, cv2.LINE_AA)
        cv2.putText(result, "Car offset(m): {0:0.2f}".format(vehicle_offset),
                    (20,130), font, 2,(255,255,0),2, cv2.LINE_AA)
    return result, left_fit, right_fit


# Helper function to do all these steps:
# 1. Distortion correction
# 2. Color/gradient threshold
# 3. Perspective transform
# 4. Detect lane lines
# 5. Determine the lane curvature and draw lane area.
def process_image(image, left_fit_prev=None, right_fit_prev=None):
    warped, _, Minv, _ = perspective_transform(undistort_and_threshold(image))
    left_fit, right_fit = polyfit_from_previous(warped, left_fit_prev, right_fit_prev)
    if left_fit is None or right_fit is None:
        left_fit, right_fit = sliding_window_poly_fit(warped, visualize=False)
    if left_fit is not None and right_fit is not None:
        return draw_lane_area(image, warped, Minv, left_fit, right_fit)
    return image, left_fit, right_fit
```


```python
img = mpimg.imread('test_images/test6.jpg')
result,_,_ = process_image(img)
plt.imshow(result)
plt.show()
```


![png](writeup_images/output_29_0.png)


### Pipeline (video)

### Rubric point 8
#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

https://youtu.be/ER6T27KdeEE


```python
from moviepy.editor import VideoFileClip

class Lanes():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None

def process_video(input_path, output_path):
    """
    Process input video frame by frame to draw lane area with curvature, vehicle offset information to generate
    output video.
    """
    # Read video.
    video = VideoFileClip(input_path)
    lanes = Lanes()
    # Save left_fit, right_fit from previous processed frame.
    def process(image):
        output, left_fit, right_fit = process_image(image, lanes.left_fit, lanes.right_fit)
        lanes.left_fit = left_fit
        lanes.right_fit = right_fit
        return output

    clip = video.fl_image(process)
    clip.write_videofile(output_path, audio=False)

# video
process_video('project_video.mp4', 'output.mp4')
#process_video('challenge_video.mp4', 'challenge_output.mp4')
#process_video('harder_challenge_video.mp4', 'harder_challenge_output.mp4')
```

    [MoviePy] >>>> Building video harder_challenge_output.mp4
    [MoviePy] Writing video harder_challenge_output.mp4


    100%|█████████▉| 1199/1200 [02:05<00:00,  9.36it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: harder_challenge_output.mp4 
    


### Discussion
### Rubric point 9

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
Most time consuming part of the pipeline is thresholding an image. Figuring out right combination of channels (in image) and sobel operators to discard unnecessary pixels which are not part of lane lines was challenging. 

As you can see in 'challenge_output.mp4' (generated from challenge_video.mp4), my pipeline failed to identify lane lines because the lane is damaged slightly (with a line in the middle). My implementation assumes that there will be only two lines in the warped image. In the challenge video, thresolded warped image will have three 3 lines. Histogram peak method i used to find base of left and right lines will not work because of new line in the middle.

There are many ways to solve this issue. I will mention one simple method that can be used to solve this issue: 
1. Find the base of the line in the middle and traverse up to find the lane and discard all the pixels near that polyfit lane
2. Run the orginal histogram peak method to identify lanes.
