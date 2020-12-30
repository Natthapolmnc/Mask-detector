import cv2 as cv

image=cv.imread("data_preprocess/train/train_image_070.jpg")
# drw_img=cv.rectangle(image,(35,347),(35+383,347+289),(255,255,0))
drw_img=cv.resize(image,(256,256))
cv.imshow("kai",drw_img)
cv.waitKey()