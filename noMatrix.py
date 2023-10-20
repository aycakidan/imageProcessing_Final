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

# Verify that the images have been successfully uploaded
if image1 is None or image2 is None or image3 is None or image4 is None:
    print("One or more images could not be uploaded.")
else:
    # Image processing and matching operations
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

    # Use the SIFT algorithm for image processing and matching
    sift = cv2.SIFT_create()

    # Keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)
    keypoints4, descriptors4 = sift.detectAndCompute(gray4, None)

# FLANN-based matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Matching keypoints
matches1 = flann.knnMatch(descriptors1, descriptors2, k=2)
matches2 = flann.knnMatch(descriptors1, descriptors3, k=2)
matches3 = flann.knnMatch(descriptors1, descriptors4, k=2)

# Filtering matching points and selecting top 100 matches
good_matches1 = []
for m, n in matches1:
    if m.distance < 0.6 * n.distance:
        good_matches1.append(m)
good_matches1 = sorted(good_matches1, key=lambda x: x.distance)[:100]

good_matches2 = []
for m, n in matches2:
    if m.distance < 0.6 * n.distance:
        good_matches2.append(m)
good_matches2 = sorted(good_matches2, key=lambda x: x.distance)[:100]

good_matches3 = []
for m, n in matches3:
    if m.distance < 0.6 * n.distance:
        good_matches3.append(m)
good_matches3 = sorted(good_matches3, key=lambda x: x.distance)[:100]

    # Visualizing matching points
img_matches1 = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                   good_matches1, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches2 = cv2.drawMatches(image1, keypoints1, image3, keypoints3,
                                   good_matches2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches3 = cv2.drawMatches(image1, keypoints1, image4, keypoints4, good_matches3, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show matching points
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img_matches1, cv2.COLOR_BGR2RGB))
axes[0].axis('off')
axes[0].set_title('Matches between Image 1 and Image 2')

axes[1].imshow(cv2.cvtColor(img_matches2, cv2.COLOR_BGR2RGB))
axes[1].axis('off')
axes[1].set_title('Matches between Image 1 and Image 3')

plt.tight_layout()
plt.show()
