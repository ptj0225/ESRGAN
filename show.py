import matplotlib.pyplot as plt
import cv2
import time

cv2.namedWindow("preview")

for i in range(100):
    img_lr = cv2.imread(f"./test/{i}.png")
    img_sr = cv2.imread(f"./result/{i}.png")
    cv2.imshow('preview',img_sr)
    cv2.waitKey(0)