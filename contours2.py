import cv2
from imutils import contours as imutils_contours
import numpy as np
import math
import os

def order_points(points):
    """
        Credits to: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    :param points: List of 4 points
    :return:
    """
    pts = np.array([item for sublist in points for item in sublist])
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def distance_between_points(point1, point2):
    return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))


def get_new_image_size_from_points(points):
    top_left, top_right, bottom_right, bottom_left = points

    print(top_left)
    print(top_right)
    print(bottom_right)
    print(bottom_left)

    top_width = int(distance_between_points(top_left, top_right))
    bottom_width = int(distance_between_points(bottom_left, bottom_right))
    max_width = max(top_width, bottom_width)

    left_height = int(distance_between_points(top_left, bottom_left))
    right_height = int(distance_between_points(top_right, bottom_right))
    max_height = max(left_height, right_height)

    return max_width, max_height


image = cv2.imread('sudoku9.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
threshold = 255 - threshold

contours, hierarchies = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f'Found {len(contours)} contour(s).')

blank = np.zeros(image.shape[:2], dtype='uint8')
# cv.drawContours(blank, contours, -1, (255, 255, 255), -1)


max_area = 0
best_cnt = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

cv2.drawContours(blank, best_cnt, -1, (255, 255, 255), 1)
mask = np.zeros((image.shape[:2]), np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)

# Must reorder points in correct position using order_points function!
corners = cv2.goodFeaturesToTrack(mask, 4, 0.1, 10)
corners = order_points(corners)
width, height = get_new_image_size_from_points(corners)
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]], dtype="float32")

perspective_matrix = cv2.getPerspectiveTransform(corners, dst)
warped = cv2.warpPerspective(image, perspective_matrix, (width, height))
grid = np.copy(warped)
cv2.imwrite(f'./warped.png', warped)

cv2.imshow('warped', warped)

# Load image, grayscale, and adaptive threshold
image = cv2.imread('warped.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

cv2.imshow('thresh', thresh)

# Filter out all numbers and noise to isolate only boxes
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

# Fix horizontal and vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

# Sort by top to bottom and each row by left to right
invert = 255 - thresh
cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = imutils_contours.sort_contours(cnts, method="top-to-bottom")

sudoku_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    area = cv2.contourArea(c)
    if area < 50000:
        row.append(c)
        if i % 9 == 0:
            (cnts, _) = imutils_contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(cnts)
            row = []

# Iterate through each box
i = 1
for row in sudoku_rows:
    j = 1
    for col in row:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [col], -1, (255, 255, 255), 1)
        print(np.sum(mask))
        result = cv2.bitwise_and(image, mask)
        result[mask == 0] = 255
        cv2.imshow('cell', mask)
        cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(col)
        result = image[y:y + h, x:x + w]
        inverted_result = 255 - result
        inverted_result = cv2.resize(inverted_result, (28, 28))

        if not cv2.imwrite(f'..\\images from sudoku\\Try 3\\cell{i}-{j}.jpg', inverted_result):
            raise Exception("Could not write image")
        j += 1
    i += 1
