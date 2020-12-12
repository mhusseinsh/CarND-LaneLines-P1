# **Finding Lane Lines on the Road**

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./output_images_split/solidWhiteRight_input.jpg "Input Image"
[image2]: ./output_images_split/solidWhiteRight_gray.jpg "Grayscale"
[image3]: ./output_images_split/solidWhiteRight_blur.jpg "Blur"
[image4]: ./output_images_split/solidWhiteRight_edges.jpg "Edges"
[image5]: ./output_images_split/solidWhiteRight_roi.jpg "Target"
[image6]: ./output_images_split/solidWhiteRight_lines.jpg "Lines"
[image7]: ./output_images_split/solidWhiteRight_result.jpg "Result"
[image8]: ./output_images/solidWhiteRight_result.jpg "ResultFull"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

Overview:
Lane detection is an important foundation in the development of intelligent vehicles. To reach levels of autonomy, this algorithm need to be implemented in the vehicle to be able for the car to know where exactly it is driving.

This project aims to find lanes using classical opencv approaches. To find lanes, we will use [Canny Edge Detection](https://docs.opencv.org/master/da/d22/tutorial_py_canny.html) to detect edges in a pre-processed gray image. After getting the edges, [Hough Line Transformation](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html) will be used to find lines which we will overlay on the image.
![alt text][image1]
My pipeline consisted of several steps:

1- Converting the image to grayscale using the [cvtColor()](https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab) method.
![alt text][image2]
2- Before applying the Canny algorithm, we need to smoothen the image to be able to supress the noise. Here, [Gaussian Blur](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1) with kernel size of 5 is used.
![alt text][image3]
3- To detect the edges of the lanes, Canny Edge Detector is used. This algorithm works by using the pixel gradient values then filters the provided image according to the Lower and Higher threshold which are two passed arguments. All pixels which lie below this lower threshold are ignored, while others above the higher threshold are accepted. Moreover, the pixels which lie in between the lower and the higher threholds are accepted if they connect with the pixels above the higher threshold.
The ration between thresholds which are normally used are 1:2 or 1:3, in this case, I used 1:2, with a values of 50 and 150.
![alt text][image4]
4- Since normally the lines lie in the lower center part of the image, then choosing a region of interest will help in identifying the lines easier and gives a higher probability in this task. Accordingly, a Region of Interest (ROI) with of a trapeziodal polygon shape is chosen. The area which lie outside this ROI is excluded.
![alt text][image5]
5- Afterwards the Hough Line Transform is applied to be able to find the lines which is detected in the corresponding ROI.
![alt text][image6]
6- To be able to draw the lines on the image and produce the final result, [addWeighted()](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19) is used to take the blank image (which is the Hough Transform output) with only lines on it, and a copy of the original image. Then it overlays the lines on the original image and returns the output.
![alt text][image7]
In order to draw a single line on the left and right lanes, the draw_lines() function is modified. There are multiple lines that can be detected for a lane line (especially if they are separated in the street image). In order to overcome this problem, it is needed to work on an averaged line method for that.
The task is to simply extrapolate all the lines to cover full lane line length for partially recognized lane lines.
As a final output, two lane lines should be drawn: one for the left and the other for the right. The left line has a negative slope, while the right one should have a positive slope. Hence, the slopes for the lines should be computed, and then separated and collected. The positive slopes will be stored separately from the negative ones, and afterwards their averages will be taken and extrapolation will be calculated.
![alt text][image8]
Videos:
![](./test_videos_output/solidWhiteRight.mp4)
<video width="960" height="540" controls>
  <source src="./test_videos_output/solidWhiteRight.mp4" type="video/mp4">
</video>
### 2. Identify potential shortcomings with your current pipeline

There are potential shortcomings with this task:

1- Tuning parameters is considered to be very tricky for this task. Choosing the correct parameters for the Hough Line Transform is not always easy. Finding the correct trade-off between the 4 parameters need to be further investigated. "Trial and Error" to retrieve the hyper-parameters is not the best way to do so.

2- Due to the reason that Hough Lines are based on straight lines do not work good for curved lines, so having such curve lines in the image, will not be easily translated to detected straight lines (This is clearly shown in the challenging video).

3- The ROI may not always be fixed. An assumption was made about the position of the lines in the image, which may not always be valid. This assumption is due that the camera is always mounted/fixed in the same position and lanes are flat.

4- In more challenging scenes, portions from the lines may be disappearing, which simply makes it even harder for this pipeline to keep being stable (This is clearly shown in the challenging video).

5- One more shortcoming would be the uneven color of the road surface in parts of the scenes, where there are typically some tire markings which causes change in color surface (This is clearly shown in the challenging video).

6- There are some roads in general which do not detect any lane markings, however, finding the ego lane is important. This means that our pipeline will not work in such scenarios.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be definetly having an optimization search method to discover the parameters and choose the best set based on some KPI values. Another improvement could be having a dynamic ROI which changes according to some features inside the image, either road curvature, or speed of the vehicle or even road slopes. Moreover, using Color Selection algortihm along with the current pipeline will aid in lanes detection and help improving the results thus avoiding or fitting to curvy lines. Furthermore, it can be considered to collect a mean of n last detected lines. This will be helpful to be able to calculate the next line, so it will help in avoiding the errors
