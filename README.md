
# Advanced Lane Finding

## Goal
Write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. Rubric points (https://review.udacity.com/#!/rubrics/571/view) are also covered in this ipython notebook.

## Pipeline
Steps involved in the pipeline:

    1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2. Apply a distortion correction to raw images.
    3. Use color transforms, gradients, etc., to create a thresholded binary image.
    4. Apply a perspective transform to rectify binary image ("birds-eye view").
    5. Detect lane pixels and fit to find the lane boundary.
    6. Determine the curvature of the lane and vehicle position with respect to center.
    7. Warp the detected lane boundaries back onto the original image.
    8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import glob

# Prepare object points
nx = 9
ny = 6

# Arrays to store objects points and image points from all the images.
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Read in and make a list of calibration images.
images = glob.glob('camera_cal/calibration*.jpg')

# Subplots initialization (https://matplotlib.org/examples/pylab_examples/subplots_demo.html).
f, axarr2D = plt.subplots(6, 3, figsize=(15,15))
f.subplots_adjust(hspace = .2, wspace=.001)
axarr1D = axarr2D.ravel()

successfully_calibrated = 0
for fname in images:
    # Read each image
    img = mpimg.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        axarr1D[successfully_calibrated].imshow(img)
        axarr1D[successfully_calibrated].axis('off')
        successfully_calibrated =  successfully_calibrated + 1

plt.show()
```


![png](output_2_0.png)


### Rubric point 1

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I used chessboard calibration images (of size 9x6) provided as part of project resources to compute camera matrix and distortion coefficients.
Steps involved:

        1. Every calibrated image is converted to grayscale
        2. cv2.findChessboardCorners is used to find corners.
        3. If successful, the corners are saved in imgpoints list.
        4. As we know the size of chessboard, objpoints list is easy to construct by generating sequence ((0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)) for each image.
        
I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function (Note: undistorted image is displayed belowe the code).


```python
# Read a test chessboard image
img = mpimg.imread('camera_cal/calibration3.jpg')

# Use image points and object points computed from chessboard calibration images
# to compute the calibration and later undistortion.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

# Helper function to undistort images.
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Helper function to undistort the original image and compare it with original side-by-side.
def undistort_and_compare(img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistort(img))
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

undistort_and_compare(img)
```


![png](output_4_0.png)


### Rubric point 2

#### Provide an example of a distortion-corrected image.
Note the change in white car's location :)


```python
# Read a test image
img = mpimg.imread('test_images/test4.jpg')
undistort_and_compare(img)
```


![png](output_6_0.png)


### Color and Thresholds

Let's see which color channels and thresholding methods are useful for identifying lanes (white and yellow color).


```python
# Function that applies Sobel x or y, then takes an absolute value
# and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Function that applies Sobel x and y, then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
```


```python
# test4.jpg is good candidate to check which color channel works better than other in detecting lane lines
img = mpimg.imread('test_images/test4.jpg')
plt.imshow(img)
plt.show()

def display_channels(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H1 = HLS[:,:,0]
    L = HLS[:,:,1]
    S1 = HLS[:,:,2]
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H2 = HSV[:,:,0]
    S2 = HSV[:,:,1]
    V = HSV[:,:,2]
    fn, axes2D = plt.subplots(3, 3, figsize=(12, 12))
    fn.tight_layout()

    axes2D[0][0].set_title('Red Channel (RGB)', fontsize=10)
    axes2D[0][0].imshow(R, cmap='gray')
    axes2D[0][1].imshow(G, cmap='gray')
    axes2D[0][1].set_title('Green Channel (RGB)', fontsize=10)
    axes2D[0][2].imshow(B, cmap='gray')
    axes2D[0][2].set_title('Blue Channel (RGB)', fontsize=10)
    axes2D[1][0].imshow(H1, cmap='gray')
    axes2D[1][0].set_title('Hue Channel (HLS)', fontsize=10)
    axes2D[1][1].imshow(L, cmap='gray')
    axes2D[1][1].set_title('Lighting Channel (HLS)', fontsize=10)
    axes2D[1][2].imshow(S1, cmap='gray')
    axes2D[1][2].set_title('Saturation Channel (HLS)', fontsize=10)
    axes2D[2][0].imshow(H2, cmap='gray')
    axes2D[2][0].set_title('Hue Channel (HSV)', fontsize=10)
    axes2D[2][1].imshow(S2, cmap='gray')
    axes2D[2][1].set_title('Saturation Channel (HSV)', fontsize=10)
    axes2D[2][2].imshow(V, cmap='gray')
    axes2D[2][2].set_title('Value Channel (HSV)', fontsize=10)
    plt.show()

display_channels(img)

# Choose a Sobel kernel size
ksize = 3

# Lets see how different thresholding methods work on the test image.
def sobel_thresholds(img):
    fn, axes2D = plt.subplots(2, 2, figsize=(12, 12))
    fn.tight_layout()

    axes2D[0][0].set_title('Absolute sobel x', fontsize=10)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    axes2D[0][0].imshow(gradx, cmap='gray')
    
    axes2D[0][1].set_title('Absolute sobel y', fontsize=10)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(50, 150))
    axes2D[0][1].imshow(grady, cmap='gray')

    axes2D[1][0].set_title('Magnitude threshold', fontsize=10)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 150))
    axes2D[1][0].imshow(mag_binary, cmap='gray')

    axes2D[1][1].set_title('Directional threshold', fontsize=10)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.02, 0.3))
    axes2D[1][1].imshow(dir_binary, cmap='gray')

sobel_thresholds(img)
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)


### Rubric point 3

#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

As you can see above, "Saturation" (higher threshold) and "Hue" (lower threshold) channel of HSV image detects lane lines better than others. Also, absolute sobel threshold X method seem to identify lanes better than other thresholding methods.

Together with HSV color transform (H and S channel) and Sobel threshold X gradient, we obtain thresholded binary image.

Directional and Magnitude thresholds has very minimal to no effect on thresholded binary image.



```python
def undistort_and_threshold(img, h_thresh=(15, 100), s_thresh=(100, 255), sx_thresh=(50, 150)):
    # Undistort the image
    img = np.copy(undistort(img))
    
    sx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]

    # Threshold color channel (hue and saturation)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((h_binary == 1) & (s_binary == 1)) | (sx_binary == 1)] = 1
    return combined_binary

img = mpimg.imread('test_images/test4.jpg')
result = undistort_and_threshold(img)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(img)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result, cmap='gray')
ax2.set_title('Thresholded Binary Image', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_11_0.png)


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


![png](output_14_0.png)


Let's apply perspective transformation on thresolded image.


```python
img = mpimg.imread('test_images/test4.jpg')
display_unwarped(undistort_and_threshold(img), cmap='gray')
```


![png](output_16_0.png)


Now let's see how transformation looks like on test images


```python
images = glob.glob('test_images/*.jpg')
for fname in images:
    img = mpimg.imread(fname)
    display_unwarped(undistort_and_threshold(img), cmap='gray')
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)



![png](output_18_5.png)



![png](output_18_6.png)



![png](output_18_7.png)


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


![png](output_21_0.png)



![png](output_21_1.png)


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


![png](output_29_0.png)


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
