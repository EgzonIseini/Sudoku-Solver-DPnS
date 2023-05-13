import cv2 as cv
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import sudoku as sudoku_solver
import time
from keras.models import load_model
from imutils import contours as imutil_contours


def get_corners_from_contour(contour):
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))

    return np.array([contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]], dtype="float32")


def distance_between_points(point1, point2):
    return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))


def get_new_image_size_from_points(points):
    top_left, top_right, bottom_right, bottom_left = np.array(points).tolist()

    top_width = int(distance_between_points(top_left, top_right))
    bottom_width = int(distance_between_points(bottom_left, bottom_right))
    max_width = max(top_width, bottom_width)

    left_height = int(distance_between_points(top_left, bottom_left))
    right_height = int(distance_between_points(top_right, bottom_right))
    max_height = max(left_height, right_height)

    return max_width, max_height

def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

start = time.time()

image = cv.imread('sudoku.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(image, (9, 9), cv.BORDER_DEFAULT)

threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
threshold = cv.bitwise_not(threshold, threshold)

contours, hierarchies = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

corners = get_corners_from_contour(best_cnt)
width, height = get_new_image_size_from_points(np.array(corners, dtype="float32"))
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]], dtype="float32")

perspective_matrix = cv.getPerspectiveTransform(corners, dst)
image = cv.warpPerspective(image, perspective_matrix, (width, height))
threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 7)
threshold = cv.bitwise_not(threshold, threshold)

cv.imshow('threshold', threshold)
cv.waitKey(0)

# Filter out all numbers and noise to isolate only boxes
cnts = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv.contourArea(c)
    if area < 1000:
        cv.drawContours(threshold, [c], -1, (0,0,0), -1)

# Fix horizontal and vertical lines
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,11))
threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, vertical_kernel, iterations=2)
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (11,1))
threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, horizontal_kernel, iterations=2)

# Sort by top to bottom and each row by left to right
invert = 255 - threshold
cnts = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = imutil_contours.sort_contours(cnts, method="top-to-bottom")


sudoku_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    area = cv.contourArea(c)
    if area < 50000:
        row.append(c)
        if i % 9 == 0:
            (cnts, _) = imutil_contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(cnts)
            row = []

sudoku = []
digits_to_recognize = []

for row in sudoku_rows:
    for c in row:
        corners = get_corners_from_contour(c).astype('uint')
        top_left = corners[0]
        bottom_right = corners[2]
        digit = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        digit = cv.resize(digit, (28, 28))
        digit = cv.adaptiveThreshold(digit, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 21)
        digit = cv.bitwise_not(digit, digit)

        white_pixels = np.count_nonzero(digit)
        if white_pixels > 35:
            digits_to_recognize.append(digit)
            sudoku.append(None)
        else:
            sudoku.append(".")



input = np.array(digits_to_recognize)
th, input = cv.threshold(input, 50, 255, cv.THRESH_BINARY)
input = np.expand_dims(input, axis=3)/255
model = load_model("../model/ocr_model.h5")
prediction = model.predict(input)
prediction = np.argmax(prediction, axis=1)

ncolumns = 4
nrows = math.ceil(len(digits_to_recognize) / ncolumns)

fig, ax = plt.subplots(nrows, ncolumns, sharex=True, sharey=True)
fig.set_figheight(12)
fig.set_figwidth(12)
for row in range(nrows):
    for col in range(ncolumns):
        i = (row * 4) + col
        if i == len(digits_to_recognize):
            break
        th, im_th = cv.threshold(digits_to_recognize[i], 50, 255, cv.THRESH_BINARY)
        im_th = cv.cvtColor(im_th, cv.COLOR_RGB2BGR)
        ax[row, col].set_title(
            "Predicted label :{}".format(prediction[i]))
        ax[row, col].imshow(im_th)

plt.show()

predictions = prediction.tolist()
for (i, number) in enumerate(sudoku):
    if number == None:
        sudoku[i] = str(predictions.pop(0))
    else:
        continue

solved_sudoku = sudoku_solver.solve(sudoku).values()

print(solved_sudoku)

for row in sudoku_rows:
    for c in row:
        corners = get_corners_from_contour(c).astype('uint')
        top_left = corners[0]
        bottom_right = corners[2]
        digit = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        digit = cv.resize(digit, (28, 28))
        digit = cv.adaptiveThreshold(digit, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 21)
        digit = cv.bitwise_not(digit, digit)

        white_pixels = np.count_nonzero(digit)
        if white_pixels > 35:
            digits_to_recognize.append(digit)
            sudoku.append(None)
        else:
            sudoku.append(".")

#end = time.time()
#print(f'Solved sudoku in {end - start} start to finish')