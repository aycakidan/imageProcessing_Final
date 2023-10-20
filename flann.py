import cv2
import matplotlib.pyplot as plt

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

# Create FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors using FLANN
matches1 = flann.knnMatch(train_descriptor1, test_descriptor, k=2)
matches2 = flann.knnMatch(train_descriptor2, test_descriptor, k=2)
matches3 = flann.knnMatch(train_descriptor3, test_descriptor, k=2)
matches4 = flann.knnMatch(train_descriptor4, test_descriptor, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches1 = []
for m, n in matches1:
    if m.distance < 0.7 * n.distance:
        good_matches1.append(m)

good_matches2 = []
for m, n in matches2:
    if m.distance < 0.7 * n.distance:
        good_matches2.append(m)

good_matches3 = []
for m, n in matches3:
    if m.distance < 0.7 * n.distance:
        good_matches3.append(m)

good_matches4 = []
for m, n in matches4:
    if m.distance < 0.7 * n.distance:
        good_matches4.append(m)

# Sort matches by distance
good_matches1 = sorted(good_matches1, key=lambda x: x.distance)
good_matches2 = sorted(good_matches2, key=lambda x: x.distance)
good_matches3 = sorted(good_matches3, key=lambda x: x.distance)
good_matches4 = sorted(good_matches4, key=lambda x: x.distance)

# Select the top 100 matches
good_matches1 = good_matches1[:100]
good_matches2 = good_matches2[:100]
good_matches3 = good_matches3[:100]
good_matches4 = good_matches4[:100]

# Draw the matches
result1 = cv2.drawMatches(image1, train_keypoints1, test_image, test_keypoints, good_matches1, test_gray, flags=2)
result2 = cv2.drawMatches(image2, train_keypoints2, test_image, test_keypoints, good_matches2, test_gray, flags=2)
result3 = cv2.drawMatches(image3, train_keypoints3, test_image, test_keypoints, good_matches3, test_gray, flags=2)
result4 = cv2.drawMatches(image4, train_keypoints4, test_image, test_keypoints, good_matches4, test_gray, flags=2)

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
print("\nNumber of Matching Keypoints Between Training Image 1 and Query Image: ", len(good_matches1))
print("Number of Matching Keypoints Between Training Image 2 and Query Image: ", len(good_matches2))
print("Number of Matching Keypoints Between Training Image 3 and Query Image: ", len(good_matches3))
print("Number of Matching Keypoints Between Training Image 4 and Query Image: ", len(good_matches4))

plt.show()
