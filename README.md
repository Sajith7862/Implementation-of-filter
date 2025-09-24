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
### Orginal image :
<img width="717" height="521" alt="image" src="https://github.com/user-attachments/assets/42c1397a-a934-4a41-8d9b-29798c8d7ed8" />

### Using Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = r"saji.png"  # Change this to your correct path

if not os.path.exists(img_path):
    print(" Image not found. Check the file path.")
else:
    image1 = cv2.imread(img_path)
    if image1 is None:
        print(" Image could not be loaded (possibly corrupted or unsupported format).")
    else:
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        kernel = np.ones((11, 11), np.float32) / 169
        image3 = cv2.filter2D(image2, -1, kernel)

        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(image2)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image3)
        plt.title("Average Filter Image")
        plt.axis("off")
        plt.show()


```
<img width="717" height="521" alt="image" src="https://github.com/user-attachments/assets/bcac4163-3c37-4264-8ac8-575869a771da" />


### Using Weighted Averaging Filter
```Python

kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16

image3=cv2.filter2D(image2,-1,kernel1)

plt.imshow(image3)

plt.title("Weighted Average Filter Image")

plt.axis("off")

plt.show()
```
<img width="275" height="411" alt="image" src="https://github.com/user-attachments/assets/73395f3b-9f2b-4d6a-95dd-a8375dc6fccd" />


### Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
<img width="264" height="411" alt="image" src="https://github.com/user-attachments/assets/9921064a-6045-45d5-9e6d-bd83e3d640b1" />


### Using Median Filter
```Python
median = cv2.medianBlur(image2, 13)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title("Median Blur")
plt.axis("off")
plt.show()

```
<img width="264" height="411" alt="image" src="https://github.com/user-attachments/assets/d11b613a-d7ee-41c5-9ac0-13c150703f61" />


### Sharpening Filters
## Using Laplacian Linear Kernal
```Python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```
<img width="264" height="411" alt="image" src="https://github.com/user-attachments/assets/83ea9c08-ad88-42e4-beba-556c77ca284b" />


### Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```
<img width="264" height="411" alt="image" src="https://github.com/user-attachments/assets/581f9e45-f90c-436d-843b-ecb7e485eff3" />

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
