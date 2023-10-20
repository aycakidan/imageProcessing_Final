import cv2
import numpy as np

img_path = r'C:\Users\AYÇA\Desktop\imageProcessing_Final\karekose.jpg'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


corners = cv2.cornerHarris(gray, 2, 3, 0.004)
corners = cv2.dilate(corners, None)
ret, corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
corners = np.uint8(corners)


corners = cv2.goodFeaturesToTrack(corners, 100, 0.01, 10)
corners = np.int0(corners)


sorted_corners = sorted(
    corners, key=lambda corner: (corner[0][1], corner[0][0]))


for i, corner in enumerate(sorted_corners, start=1):
    x, y = corner.ravel()
    print(f"Köşe {i}: x={x}, y={y}")

    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(image, str(i), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Köşeleri Tespit Et", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
