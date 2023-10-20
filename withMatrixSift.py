import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\1.jpg'
img1_path = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\2.jpg'
img2_path = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\3.jpg'
img3_path = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\4.jpg'

image1 = cv2.imread(img_path)
image2 = cv2.imread(img1_path)
image3 = cv2.imread(img2_path)
image4 = cv2.imread(img3_path)

# To verify that the images have been successfully uploaded
if image1 is None or image2 is None or image3 is None or image4 is None:
    print("Image cannot load.")
else:
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)   

# Create test image by adding Scale Invariance and Rotational Invariance
test_image = cv2.pyrDown(image1)
test_image = cv2.pyrDown(image2)
test_image = cv2.pyrDown(image3)
test_image = cv2.pyrDown(image4)
num_rows, num_cols = test_image.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# Display training images and testing image
fig, plots = plt.subplots(2, 2, figsize=(20, 10))

plots[0, 0].set_title("Training Image 1")
plots[0, 0].imshow(image1)

plots[0, 1].set_title("Training Image 2")
plots[0, 1].imshow(image2)

plots[1, 0].set_title("Training Image 3")
plots[1, 0].imshow(image3)

plots[1, 1].set_title("Testing Image")
plots[1, 1].imshow(test_image)


sift = cv2.SIFT_create()

train_keypoints1, train_descriptor1 = sift.detectAndCompute(gray1, None)
train_keypoints2, train_descriptor2 = sift.detectAndCompute(gray2, None)
train_keypoints3, train_descriptor3 = sift.detectAndCompute(gray3, None)
train_keypoints4, train_descriptor4 = sift.detectAndCompute(gray3, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

keypoints_without_size1 = np.copy(image1)
keypoints_with_size1 = np.copy(image1)

keypoints_without_size2 = np.copy(image2)
keypoints_with_size2 = np.copy(image2)

keypoints_without_size3 = np.copy(image3)
keypoints_with_size3 = np.copy(image3)

keypoints_without_size4 = np.copy(image4)
keypoints_with_size4 = np.copy(image4)

cv2.drawKeypoints(image1, train_keypoints1,
                  keypoints_without_size1, color=(0, 255, 0))
cv2.drawKeypoints(image1, train_keypoints1, keypoints_with_size1,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.drawKeypoints(image2, train_keypoints2,
                  keypoints_without_size2, color=(0, 255, 0))
cv2.drawKeypoints(image2, train_keypoints2, keypoints_with_size2,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.drawKeypoints(image3, train_keypoints3,
                  keypoints_without_size3, color=(0, 255, 0))
cv2.drawKeypoints(image3, train_keypoints3, keypoints_with_size3,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.drawKeypoints(image4, train_keypoints4,
                  keypoints_without_size4, color=(0, 255, 0))
cv2.drawKeypoints(image4, train_keypoints4, keypoints_with_size4,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display images with and without keypoints size
fig, plots = plt.subplots(4, 2, figsize=(20, 20))

plots[0, 0].set_title("Train keypoints With Size (Image 1)")
plots[0, 0].imshow(keypoints_with_size1, cmap='gray')

plots[0, 1].set_title("Train keypoints Without Size (Image 1)")
plots[0, 1].imshow(keypoints_without_size1, cmap='gray')

plots[1, 0].set_title("Train keypoints With Size (Image 2)")
plots[1, 0].imshow(keypoints_with_size2, cmap='gray')

plots[1, 1].set_title("Train keypoints Without Size (Image 2)")
plots[1, 1].imshow(keypoints_without_size2, cmap='gray')

plots[2, 0].set_title("Train keypoints With Size (Image 3)")
plots[2, 0].imshow(keypoints_with_size3, cmap='gray')

plots[2, 1].set_title("Train keypoints Without Size (Image 3)")
plots[2, 1].imshow(keypoints_without_size3, cmap='gray')

plots[3, 0].set_title("Train keypoints With Size (Image 4)")
plots[3, 0].imshow(keypoints_with_size4, cmap='gray')

plots[3, 1].set_title("Train keypoints Without Size (Image 4)")
plots[3, 1].imshow(keypoints_without_size4, cmap='gray')

# Print the number of keypoints detected in the training images
print("Number of Keypoints Detected In Training Image 1: ", len(train_keypoints1))
print("Number of Keypoints Detected In Training Image 2: ", len(train_keypoints2))
print("Number of Keypoints Detected In Training Image 3: ", len(train_keypoints3))
print("Number of Keypoints Detected In Training Image 4: ", len(train_keypoints4))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

# Create a Brute Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Perform the matching between the SIFT descriptors of the training images and the test image
matches1 = bf.match(train_descriptor1, test_descriptor)
matches2 = bf.match(train_descriptor2, test_descriptor)
matches3 = bf.match(train_descriptor3, test_descriptor)
matches4 = bf.match(train_descriptor4, test_descriptor)

# Sort the matches by distance
matches1 = sorted(matches1, key=lambda x: x.distance)
matches2 = sorted(matches2, key=lambda x: x.distance)
matches3 = sorted(matches3, key=lambda x: x.distance)
matches4 = sorted(matches4, key=lambda x: x.distance)

# Select the top 100 matches
matches1 = matches1[:100]
matches2 = matches2[:100]
matches3 = matches3[:100]
matches4 = matches4[:100]

result1 = cv2.drawMatches(image1, train_keypoints1,
                          test_gray, test_keypoints, matches1, test_gray, flags=2)
result2 = cv2.drawMatches(image2, train_keypoints2,
                          test_gray, test_keypoints, matches2, test_gray, flags=2)
result3 = cv2.drawMatches(image3, train_keypoints3,
                          test_gray, test_keypoints, matches3, test_gray, flags=2)
result4 = cv2.drawMatches(image4, train_keypoints4,
                          test_gray, test_keypoints, matches4, test_gray, flags=2)

# Display the best matching points
fig, plots = plt.subplots(2, 2, figsize=(20, 10))

plots[0, 0].set_title('Best Matching Points (Image 1)')
plots[0, 0].imshow(result1)

plots[0, 1].set_title('Best Matching Points (Image 2)')
plots[0, 1].imshow(result2)

plots[1, 0].set_title('Best Matching Points (Image 3)')
plots[1, 0].imshow(result3)

plots[1, 1].set_title('Best Matching Points (Image 4)')
plots[1, 1].imshow(result4)



# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between Training Image 1 and Query Image: ", len(matches1))
print("Number of Matching Keypoints Between Training Image 2 and Query Image: ", len(matches2))
print("Number of Matching Keypoints Between Training Image 3 and Query Image: ", len(matches3))
print("Number of Matching Keypoints Between Training Image 4 and Query Image: ", len(matches4))

plt.show()
