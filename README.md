# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.
### Step2
Convert the image from BGR to RGB.
### Step3
Apply the required filters for the image separately.
### Step4
Plot the original and filtered image by using matplotlib.pyplot.
### Step5
End the program.

## Program:
### Developed By   : Mohamed Hameem Sajith J 
### Register Number:212223240090

## Smoothing Filters
### Using Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("Tiger.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/169
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/0803f3da-7cd5-46f4-a01b-e62635486497)

### Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/30371fc8-ee70-4799-ba6a-c4d8f952d1ae)

### Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/dc51c1e6-73a4-484a-a64c-d982a2feb06a)

### Using Median Filter
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.imshow(median)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/7b52f783-b351-4fe6-a1a5-c4dd865b12af)

### Sharpening Filters
## Using Laplacian Linear Kernal
```Python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/c6f1cba2-f9d9-43b3-875e-ccb52411f840)

### Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/f1669265-f553-438d-85eb-5636ebc0e5c6)
## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
