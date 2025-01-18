# data_process.py 数据预处理脚本
# 脚本从data/目录下读取数据，将处理后的输出保存到data_processed/目录下。

## 1. 确保目录存在
import os
os.makedirs('data_processed', exist_ok=True)

## 2. opencv读取图片data/rps/paper/paper01-000.png

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/rps/paper/paper01-000.png')

# 3. 灰度化处理保存到data_processed/gray_paper01-000.png
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('data_processed/gray_paper01-000.png', gray_img)

# 4. 旋转15度（填充白色）保存到data_processed/rotated_paper01-000.png
rows, cols = gray_img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
rotated_img = cv2.warpAffine(gray_img, M, (cols, rows), borderValue=255)
cv2.imwrite('data_processed/rotated_paper01-000.png', rotated_img)

# 5. 缩放到100*100保存到data_processed/resized_paper01-000.png
resized_img = cv2.resize(rotated_img, (100, 100))
cv2.imwrite('data_processed/resized_paper01-000.png', resized_img)