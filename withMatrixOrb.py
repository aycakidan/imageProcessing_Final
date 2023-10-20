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

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

# Create ORB object
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
train_keypoints1, train_descriptor1 = orb.detectAndCompute(gray1, None)
train_keypoints2, train_descriptor2 = orb.detectAndCompute(gray2, None)
train_keypoints3, train_descriptor3 = orb.detectAndCompute(gray3, None)
train_keypoints4, train_descriptor4 = orb.detectAndCompute(gray4, None)

# Create test image by downsampling and rotating
test_image = cv2.pyrDown(image4)
num_rows, num_cols = test_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors for test image
test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

# Create a Brute Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Perform the matching between the ORB descriptors of the training images and the test image
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

# Draw the matches
result1 = cv2.drawMatches(image1, train_keypoints1, test_gray, test_keypoints, matches1, None)
result2 = cv2.drawMatches(image2, train_keypoints2, test_gray, test_keypoints, matches2, None)
result3 = cv2.drawMatches(image3, train_keypoints3, test_gray, test_keypoints, matches3, None)
result4 = cv2.drawMatches(image4, train_keypoints4, test_gray, test_keypoints, matches4, None)

# Display the results
fig, plots = plt.subplots(2, 2, figsize=(20, 10))

plots[0, 0].set_title('Best Matching Points (Image 1)')
plots[0, 0].imshow(result1)

plots[0, 1].set_title('Best Matching Points (Image 2)')
plots[0, 1].imshow(result2)

plots[1, 0].set_title('Best Matching Points (Image 3)')
plots[1, 0].imshow(result3)

plots[1, 1].set_title('Best Matching Points (Image 4)')
plots[1, 1].imshow(result4)

plt.show()
