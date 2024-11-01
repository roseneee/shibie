import cv2
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

### 超参数表

# 标志位
blk = 1
wht = 2

# 二值化阈值区间
min_blk = 25
max_blk = 100
min_wht = 125
max_wht = 200

# 设置阈值
thresh = 150  # 判定棋子阈值

# 定义形态学变化结构元素
radius = 4  # 棋子
kernel_line = np.ones((1, 1), np.uint8)  # 线

# 高斯滤波环
ksize = 7  # 横纵线
ksize_thresh = 4  # 二值化

# 定义图片目标大小
target_size = (600, 600)







# Part 1
# 读取彩色图像
image = cv2.imread('4.png')
# cv2.imshow('Image', image)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 提取蓝色分量
blue_image = image[:, :, 0]





# Part 2
# 将灰度图转化为数组并做出直方图以便选择棋子分割阈值
# Part 3
# 在一定区域内选择直方图波谷作为分割阈值进行二值化
def threshold(img, title, min, max, sigma):
    # 将图像转换为NumPy数组
    image_array = np.array(img)

    # 获取唯一像素值及其出现次数
    unique, counts = np.unique(image_array, return_counts=True)

    # 进行高斯滤波
    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)

    # 在区间内求最小值
    start_index = np.searchsorted(unique, min)
    end_index = np.searchsorted(unique, max, side='right')

    if start_index < len(unique) and end_index <= len(unique):
        interval_counts = smoothed_counts[start_index:end_index]
        min_value = np.min(interval_counts)
        min_index = np.argmin(interval_counts)
        min_pixel_value = unique[start_index + min_index]

    # 作图
    plt.figure(figsize=(12, 6))

    # 原始直方图
    plt.subplot(1, 2, 1)
    plt.bar(unique, counts, width=1.0, edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.title(title)

    # 滤波后的直方图
    plt.subplot(1, 2, 2)
    plt.plot(unique, smoothed_counts, color='red')
    if 'min_pixel_value' in locals():
        plt.axvline(x=min_pixel_value, color='blue', linestyle='--', label=f'Min Value: {min_pixel_value}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Smoothed Count')
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(min_pixel_value)
    return min_pixel_value



# Part 4
# 根据峰值寻找纵横十九道
def line(img1, img2, ksize):
    # # 合图
    # blended_image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    blended_image = img1
    # 膨胀
    image = cv2.erode(blended_image, kernel_line, iterations=1) #  dilate
    # 高斯滤波 卷积核, 标准差为
    # image = cv2.GaussianBlur(image, (ksize, ksize), 1)
    # # 平均值滤波
    # image =  cv2.medianBlur(image, 3)

    # cv2.imshow('Closed Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 计算每一行列像素之和
    col_sum = np.sum(image, axis=0)
    row_sum = np.sum(image, axis=1)
    # 高斯滤波
    col_sum = gaussian_filter1d(col_sum, sigma=ksize)
    row_sum = gaussian_filter1d(row_sum, sigma=ksize)

    # 找到波谷
    inverted_col_sum = np.max(col_sum) - col_sum
    inverted_row_sum = np.max(row_sum) - row_sum
    valleys_col, _ = find_peaks(inverted_col_sum)
    valleys_row, _ = find_peaks(inverted_row_sum)
    # 获取波谷深度
    valley_col_depths = inverted_col_sum[valleys_col]
    valley_row_depths = inverted_row_sum[valleys_row]

    # 找到最深的19个波谷
    if len(valleys_col) >= 19:
        deepest_valleys_col_indices = np.argsort(valley_col_depths)[-19:]
        deepest_valleys_col = valleys_col[deepest_valleys_col_indices]
    else:
        print("Cannot find 19 valleys!")
        deepest_valleys_col = valleys_col

    if len(valleys_row) >= 19:
        deepest_valleys_row_indices = np.argsort(valley_row_depths)[-19:]
        deepest_valleys_row = valleys_row[deepest_valleys_row_indices]
    else:
        print("Cannot find 19 valleys!")
        deepest_valleys_row = valleys_row

    # 找到 19 行列
    col_indices = np.argsort(deepest_valleys_col)
    col = deepest_valleys_col[col_indices]
    row_indices = np.argsort(deepest_valleys_row)
    row = deepest_valleys_row[row_indices]

    # 打印最深的19个波谷位置
    print("Deepest 19 Valleys:", col)
    print("Deepest 19 Valleys:", row)

    # 绘制每列像素和的直方图和最深的波谷
    plt.figure(figsize=(10, 5))
    plt.plot(col_sum, label='Column Sum')
    plt.plot(deepest_valleys_col, col_sum[deepest_valleys_col], "o", label='Deepest Valleys')
    plt.title('Column Sum with Deepest Valleys')
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Pixels')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(row_sum, label='Row Sum')
    plt.plot(deepest_valleys_row, row_sum[deepest_valleys_row], "o", label='Deepest Valleys')
    plt.title('Column Sum with Deepest Valleys')
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Pixels')
    plt.legend()
    plt.show()

    return col, row


# Part5
# 判断点位上是否有黑白棋子
def find(img_blk, img_wht, col, row, thresh):

    # 初始化棋子矩阵
    qizi = np.zeros((19, 19), dtype=np.int32)
    # 定义卷积核（例如，3x3 全 1 矩阵）
    kernel_size = 20  # 卷积核大小
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 对图像应用卷积，计算邻域平均值
    smoothed_image_blk = cv2.filter2D(img_blk, -1, kernel)
    smoothed_image_wht = cv2.filter2D(img_wht, -1, kernel)
    # bar(smoothed_image_blk, 'blk')
    # bar(smoothed_image_wht, 'wht')

    for r in range(0, 18):
        for c in range(0, 18):
            # 获取到灰度值
            gray_blk = smoothed_image_blk[row[r], col[c]]
            gray_wht = smoothed_image_wht[row[r], col[c]]
            # print(gray_wht, gray_blk)

            if gray_blk < thresh:
                qizi[r, c] = 1

            if gray_wht < thresh:
                qizi[r, c] = -1

    print(qizi)










# 做出分布直方图
thresh_blk = threshold(gray_image, 'gray_bar', min_blk, max_blk, ksize_thresh)
thresh_wht_blue = threshold(blue_image, 'blue_bar', min_wht, max_wht, ksize_thresh)

# 棋子二值化
im_blk_board = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )  # 高斯自适应阈值二值化处理找棋盘
_, im_blk = cv2.threshold(gray_image, thresh_blk, 255, cv2.THRESH_BINARY)
_, im_wht_blue = cv2.threshold(blue_image, thresh_wht_blue, 255, cv2.THRESH_BINARY)
im_wht_blue = cv2.bitwise_not(im_wht_blue)   # 白棋子颜色取反

# 缩小图像到目标大小
im_blk = cv2.resize(im_blk, target_size, interpolation=cv2.INTER_AREA)
im_blk_board = cv2.resize(im_blk_board, target_size, interpolation=cv2.INTER_AREA)
im_wht_blue = cv2.resize(im_wht_blue, target_size, interpolation=cv2.INTER_AREA)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
# 闭运算
closed_image_blk = cv2.morphologyEx(im_blk, cv2.MORPH_CLOSE, kernel)
closed_image_wht = cv2.morphologyEx(im_wht_blue, cv2.MORPH_CLOSE, kernel)

# 缩小图像到目标大小
resized_image_blk = cv2.resize(closed_image_blk, target_size, interpolation=cv2.INTER_AREA)
resized_image_wht = cv2.resize(closed_image_wht, target_size, interpolation=cv2.INTER_AREA)




# 合并图像求纵横十九道
col, row = line(im_blk_board, im_wht_blue, ksize)
find(im_blk, im_wht_blue, col, row, thresh)


# # 显示结果
# cv2.imshow('Closed Image blk', resized_image_blk)
# cv2.imshow('Closed Image wht', resized_image_wht)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Closed Image wht', im_wht_blue)
# cv2.imshow('Closed Image blk', im_blk)
# cv2.imshow('Closed Image blk qizi', im_blk_board)
# cv2.waitKey(0)
# cv2.destroyAllWindows()