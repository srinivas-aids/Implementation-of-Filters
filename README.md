# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the necessary modules.

### Step2
For performing smoothing operation on a image.

    Average filter
```python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
    Weighted average filter
```python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
    Gaussian Blur
```python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
    Median filter
```python
median=cv2.medianBlur(image2,13)
```
### Step3
For performing sharpening on a image.

    Laplacian Kernel
```python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
    Laplacian Operator
```python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```
### Step4
Display all the images with their respective filters.
## Program:
```PYTHON
### Developed By   : SRINIVAS.U
### Register Number: 212221230108
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("img.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
plt.imshow(image1)
plt.axis("off")
plt.show()
plt.imshow(image2)
plt.axis("off")
plt.show()
```
### 1. Smoothing Filters

i) Using Averaging Filter
```Python

kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
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
ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()




```
iii) Using Gaussian Filter
```Python

gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()


```

iv) Using Median Filter
```Python

median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()



```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()




```
ii) Using Laplacian Operator
```Python

laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()




```

## OUTPUT:

### 1. Smoothing Filters

i) Using Averaging Filter


<img width="485" alt="image" src="https://user-images.githubusercontent.com/93427183/230021956-644fb70b-c8dd-4965-bfad-277bf45b3e05.png">


ii) Using Weighted Averaging Filter


<img width="526" alt="image" src="https://user-images.githubusercontent.com/93427183/230023707-4df25e1a-01bc-44d1-b39b-a799b297a77e.png">


iii) Using Gaussian Filter


<img width="501" alt="image" src="https://user-images.githubusercontent.com/93427183/230023770-c2553252-c910-4a07-9338-34732cad8e29.png">


iv) Using Median Filter

<img width="494" alt="image" src="https://user-images.githubusercontent.com/93427183/230022969-996b1a3c-a5f9-4e03-9226-139bb4cbf9cd.png">

### 2. Sharpening Filters
i) Using Laplacian Kernal

<img width="485" alt="image" src="https://user-images.githubusercontent.com/93427183/230022881-fc0beb10-9680-4f6c-917d-a84406d123c9.png">


ii) Using Laplacian Operator

<img width="490" alt="image" src="https://user-images.githubusercontent.com/93427183/230023044-5028ad1a-5302-41b3-b1fb-c3b21f8edc3b.png">


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
