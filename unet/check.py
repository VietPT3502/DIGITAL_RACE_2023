import cv2

img = cv2.imread("/home/vietpt/vietpt/vietpt/race/unet/predicted_mask.jpg")

cv2.imshow("image", img)
cv2.waitKey(0)