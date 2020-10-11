import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as pl

image_list = []
for filename in glob.glob(r"Enter your dataset address here"):
    im = cv2.imread(filename)
    image_list.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


image_array = np.array(image_list)

image_flatten = image_array.reshape(image_array.shape[0], -1).T

mean_image = np.mean(image_flatten, axis=1)
print(mean_image.shape)

# mean_image = mean_image.astype(int)
print(mean_image)

# mean_image_reshape = np.array(mean_image.reshape(218, 178)).astype(np.uint8)

#cv2.imshow("The Mean image", mean_image_reshape)
#cv2.waitKey(0)


NUM_EIGEN_FACES = 100

#MAX_SLIDER_VALUE = 255

image_flatten = image_flatten.T

print("Calculating PCA ", end="...")
mean, eigenVectors = cv2.PCACompute(image_flatten, mean=None, maxComponents=NUM_EIGEN_FACES)
print("DONE")
print("The mean size is", mean.shape)
print("The eigenVectors size is", eigenVectors.shape)

mean_image_reshape = np.array(mean.reshape(218, 178)).astype(np.uint8)

pl.imshow(mean_image_reshape, cmap="gray")
pl.show()
eigenFaces = []

for eigenVector in eigenVectors:
    eigenFace = eigenVector.reshape(218, 178).astype(np.float32)
    eigenFaces.append(eigenFace)

temp = eigenFaces[2] * 255
print(temp)

# You can mix eigenfaces together

pl.imshow(eigenFaces[2]+eigenFaces[2]+eigenFaces[5])
pl.show()

cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

























