import cv2
import matplotlib.pyplot as plt

# Görüntülerin yollarını belirtin
img_path1 = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\mouseLeft.jpg'
img_path2 = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\mouseSide.jpg'
img_path3 = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\mouseRight.jpg'

# Görüntüleri yükle
image1 = cv2.imread(img_path1)
image2 = cv2.imread(img_path2)
image3 = cv2.imread(img_path3)

# Görüntüleri gri tonlamalı hale getir
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# SIFT özellik çıkarıcısını oluştur
sift = cv2.SIFT_create()

# Keypoint'leri ve açıklama vektörlerini bul
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)

# Keypoint sayısını sınırla
max_keypoints = 100
keypoints1 = sorted(keypoints1, key=lambda x: x.response, reverse=True)[:max_keypoints]
keypoints2 = sorted(keypoints2, key=lambda x: x.response, reverse=True)[:max_keypoints]
keypoints3 = sorted(keypoints3, key=lambda x: x.response, reverse=True)[:max_keypoints]

# Açıklama vektör boyutunu sınırla
descriptor_dimension = 128
descriptors1 = descriptors1[:, :descriptor_dimension]
descriptors2 = descriptors2[:, :descriptor_dimension]
descriptors3 = descriptors3[:, :descriptor_dimension]

# FlannMatcher nesnesi oluştur
flann = cv2.FlannBasedMatcher()

# Eşleştirmeleri bul
matches1_2 = flann.knnMatch(descriptors1, descriptors2, k=2)
matches1_3 = flann.knnMatch(descriptors1, descriptors3, k=2)
matches2_3 = flann.knnMatch(descriptors2, descriptors3, k=2)

# İyi eşleştirmeleri filtrele
good_matches1_2 = []
good_matches1_3 = []
good_matches2_3 = []

for match in matches1_2:
    if len(match) == 2 and match[0].distance < 0.6 * match[1].distance:
        good_matches1_2.append(match[0])

for match in matches1_3:
    if len(match) == 2 and match[0].distance < 0.6 * match[1].distance:
        good_matches1_3.append(match[0])

for match in matches2_3:
    if len(match) == 2 and match[0].distance < 0.6 * match[1].distance:
        good_matches2_3.append(match[0])

# En az bir iyi eşleşme varsa, eşleştirmeleri görselleştir
if good_matches1_2 and good_matches1_3 and good_matches2_3:
    img_matches1_2 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches1_2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches1_3 = cv2.drawMatches(image1, keypoints1, image3, keypoints3, good_matches1_3, None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches2_3 = cv2.drawMatches(image2, keypoints2, image3, keypoints3, good_matches2_3, None, matchColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Eşleşen noktaları görselleştir
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img_matches1_2, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    axes[0].set_title('Image 1 - Image 2 Matches')
    axes[1].imshow(cv2.cvtColor(img_matches1_3, cv2.COLOR_BGR2RGB))
    axes[1].axis('off')
    axes[1].set_title('Image 1 - Image 3 Matches')
    axes[2].imshow(cv2.cvtColor(img_matches2_3, cv2.COLOR_BGR2RGB))
    axes[2].axis('off')
    axes[2].set_title('Image 2 - Image 3 Matches')
    plt.tight_layout()
    plt.show()
else:
    print('İyi eşleşme bulunamadı.')
