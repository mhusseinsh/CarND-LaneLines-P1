#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5, improveFunction=True):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/improveFunction the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and improveFunction to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    # Improve draw lines, in case of extrapolation
    if improveFunction:
        # left slope, intercept
        left_m = []
        left_b = []
        # right slope, intercept
        right_m = []
        right_b = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Line slope
                m = (y2-y1)/(x2-x1)
                
                # If slope value is infinite or not within range of interest, then we ignore it
                if (np.isfinite(m)) & (abs(m) < 1) & (abs(m) > 0.5):
                    # y-intercept
                    b = y1 - m*x1
                    
                    # Left lines have negative slope, while right lines have positive slopes
                    if m < 0:
                        left_m.append(m)
                        left_b.append(b)
                    else:
                        right_m.append(m)
                        right_b.append(b)
        
        # Averaging coefficient of left and right lines
        # Followed by extrapolation calculation
        if len(left_m) > 0:
            _m = sum(left_m)/len(left_m)
            _b = sum(left_b)/len(left_b)
            l_p0 = (int((img.shape[0]-_b)/_m), img.shape[0])
            l_p1 = (int((img.shape[0]/2*1.2-_b)/_m), int(img.shape[0]/2*1.2))
            cv2.line(img, l_p0, l_p1, color, thickness)
        if len(right_m) > 0:
            _m = sum(right_m)/len(right_m)
            _b = sum(right_b)/len(right_b)
            r_p0 = (int((img.shape[0]-_b)/_m), img.shape[0])
            r_p1 = (int((img.shape[0]/2*1.2-_b)/_m), int(img.shape[0]/2*1.2))
            cv2.line(img, r_p0, r_p1, color, thickness)
    
    # Basic line drawing based on Hough output
    else:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, improveFunction):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, improveFunction=improveFunction) # call the function above to draw lines
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def check_dir(filename):
    # check dir: checks the path of a given filename/directory, if it doesn't exist, then create the path
    #
    # filename given filename/directory to be checked
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
images = os.listdir("test_images/")
columns = 7
rows = 6
image_cell = 0
# Algorithms parameters
kernel_size = 5
low_threshold = 50
high_threshold = 150
improveFunction = True
if (improveFunction):
    output_directory = "output_images/"
    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 70     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
else:
    output_directory = "output_images_split/"
    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

check_dir(output_directory)
for idx, file in enumerate(images):
    base_name = file.split(".")[0]
    extension = file.split(".")[1]
    # RGB input image
    image = mpimg.imread("test_images/"+file)
    plt.imsave(output_directory + base_name + "_input." + extension, image, cmap='gray')

    # Convert to grayscale
    gray = grayscale(image)
    plt.imsave(output_directory + base_name + "_gray." + extension, gray, cmap='gray')

    # Apply Gaussian Blur
    blur = gaussian_blur(gray, kernel_size)
    plt.imsave(output_directory + base_name + "_blur." + extension, blur, cmap='gray')
    
    # Apply Canny Edge Detection
    edges = canny(blur, low_threshold, high_threshold)
    plt.imsave(output_directory + base_name + "_edges." + extension, edges, cmap='gray')

    # Region of Interest
    roi_ul = (100,image.shape[0])
    roi_ur = (450, 330)
    roi_lr = (520, 330)
    roi_ll = (image.shape[1],image.shape[0])
    vertices = np.array([[roi_ul, roi_ur, roi_lr, roi_ll]], dtype=np.int32)
    #plt.imsave(output_directory + base_name + "_roi." + extension, target, cmap='gray')
    target = region_of_interest(edges, vertices)
    plt.imsave(output_directory + base_name + "_target." + extension, target, cmap='gray')
    # Create figure and axes
    f, ax = plt.subplots(1, 3, figsize=(14,5))
    ax[0].imshow(edges, cmap='gray')
    ax[0].set_title('Canny Edge Detection')
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Region of Interest')
    ax[2].imshow(target, cmap='gray')
    ax[2].set_title('Target')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    x = [roi_ul[0], roi_ur[0], roi_lr[0], roi_ll[0], roi_ul[0]]
    y = [roi_ul[1], roi_ur[1], roi_lr[1], roi_ll[1], roi_ul[1]]
    ax[0].plot(x, y, 'b--', lw=2, alpha=0)
    ax[1].plot(x, y, 'b--', lw=2)
    ax[2].plot(x, y, 'b--', lw=2, alpha=0)
    #ax[1].plot(x, y, 'b--', lw=2)
    f.tight_layout()
    #f.subplots_adjust(top=0.95)
    f.savefig(output_directory + base_name + "_roi." + extension)

    # Hough Line Transformation
    lines = hough_lines(target, rho, theta, threshold, min_line_length, max_line_gap, improveFunction)
    plt.imsave(output_directory + base_name + "_lines." + extension, lines, cmap='gray')

    # Overlay image and output result
    result = weighted_img(lines, image)
    plt.imsave(output_directory + base_name + "_result." + extension, result, cmap='gray')
    
    
    #plt.figure(figsize=(25,15))
    #plt.subplot(rows, columns, 1), plt.imshow(image)
    #plt.title("origin"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 2), plt.imshow(gray, cmap='gray')
    #plt.title("gray"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 3), plt.imshow(blur, cmap='gray')
    #plt.title("blur"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 4), plt.imshow(edges, cmap='gray')
    #plt.title("edges"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 5), plt.imshow(target, cmap='gray')
    #plt.title("target"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 6), plt.imshow(lines, cmap='gray')
    #plt.title("lines"), plt.xticks([]), plt.yticks([])
    #plt.subplot(rows, columns, 7), plt.imshow(result)
    #plt.title("result"), plt.xticks([]), plt.yticks([])
    #plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # Convert to grayscale
    gray = grayscale(image)

    # Apply Gaussian Blur
    gray = gaussian_blur(gray, kernel_size)

    # Apply Canny Edge Detection
    edges = canny(gray, low_threshold, high_threshold)

    # Region of Interest
    roi_ul = (100,image.shape[0])
    roi_ur = (450, 330)
    roi_lr = (520, 330)
    roi_ll = (image.shape[1],image.shape[0])
    vertices = np.array([[roi_ul, roi_ur, roi_lr, roi_ll]], dtype=np.int32)
    target = region_of_interest(edges, vertices)

    # Hough Line Transformation
    lines = hough_lines(target, rho, theta, threshold, min_line_length, max_line_gap, improveFunction=improveFunction)

    # Overlay image and output result
    result = weighted_img(lines, image)

    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
