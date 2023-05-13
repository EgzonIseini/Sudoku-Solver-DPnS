import math
import cv2 as cv
import numpy as np
import sudoku_solver
import utils
import time

from imutils import contours as imutil_contours
from keras.models import load_model


class SudokuCell:
    def __init__(self, cell_name, cell_corners, cell_image, cell_value, is_empty):
        self.cell_name = cell_name
        self.cell_corners = cell_corners
        self.cell_image = cell_image
        self.cell_value = cell_value
        self.is_empty = is_empty

    def __str__(self):
        return f'Cell Name:{self.cell_name}\nEmpty: {self.is_empty}\n Cell corners: {self.cell_corners}\n Cell Value: {self.cell_value}'


def extract_sudoku_grid_from_image(image):
    sudoku_image = image
    sudoku_image = cv.cvtColor(sudoku_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(sudoku_image, (9, 9), cv.BORDER_DEFAULT)

    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    threshold = cv.bitwise_not(threshold, threshold)

    contours, hierarchies = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_cnt = max(contours, key=cv.contourArea)
    corners = utils.get_corners_from_contour(best_cnt)
    width, height = utils.get_new_image_size_from_points(np.array(corners, dtype="float32"))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    perspective_matrix = cv.getPerspectiveTransform(corners, dst)
    sudoku_image = cv.warpPerspective(sudoku_image, perspective_matrix, (width, height))
    sudoku_image = cv.resize(sudoku_image, (512, 512))

    return sudoku_image, corners, (width, height)


model = load_model("../model/ocr_model7.h5")
video_capture = cv.VideoCapture("http://192.168.1.102:4747/video")
cell_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

video_capture.set(cv.CAP_PROP_FPS, 10)

solved_sudoku = False
sudoku_cells = {}

target = 5
counter = 0
while (True):

    # Capture frame-by-frame

    if target == counter:
        success, image = video_capture.read()
        counter = 0

        if not success:
            continue

        original_image = np.copy(image)
        image, original_sudoku_corners, original_sudoku_size = extract_sudoku_grid_from_image(image)

        cv.imshow('Sudokuto', image)

        if solved_sudoku is False:

            threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            threshold = cv.bitwise_not(threshold, threshold)

            # Filter out all numbers and noise to isolate only boxes
            contours, hierarchies = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv.contourArea(c)
                if area < 1000:
                    cv.drawContours(threshold, [c], -1, (0, 0, 0), -1)

            # Fix horizontal and vertical lines
            vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 11))
            threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, vertical_kernel, iterations=2)
            horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 1))
            threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, horizontal_kernel, iterations=2)
            invert = np.bitwise_not(threshold)

            # Sort by top to bottom and each row by left to right
            contours, hierarchies = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            (contours, _) = imutil_contours.sort_contours(contours, method="top-to-bottom")

            row = []
            sudoku_cells = {}
            for (idx, contour) in enumerate(contours, 1):
                area = cv.contourArea(contour)
                if area < 10000:
                    row.append(contour)
                    if idx % 9 == 0:
                        (cells, _) = imutil_contours.sort_contours(row, method="left-to-right")

                        for (cell_idx, cell) in enumerate(cells, 1):

                            corners = utils.get_corners_from_contour(cell).astype('uint')
                            top_left = corners[0] + 4
                            bottom_right = corners[2] - 4
                            digit = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                            if not digit.any():
                                continue

                            digit_area = digit.shape[0] * digit.shape[1]

                            # If aspect ratio is not small, definitely not a grid cell.
                            #  print(digit.shape[1] / digit.shape[0])
                            if (digit.shape[1] / digit.shape[0]) > 3:
                                continue

                            _, digit = cv.threshold(digit, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                            row_location = math.floor(idx / 9) - 1
                            cell_name = cell_letters[row_location] + str(cell_idx)

                            sudoku_cell = SudokuCell(cell_name, corners, digit, '.', True)
                            sudoku_cells[cell_name] = sudoku_cell

                            if all(np.all(np.take(digit, index, axis=axis) == 0) for axis in range(digit.ndim) for index
                                   in
                                   (0, -1)):
                                contours, hierarchies = cv.findContours(digit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                                if len(contours) > 0:
                                    best_cnt = max(contours, key=cv.contourArea)
                                    contour_area = cv.contourArea(best_cnt)

                                    # If % of area is less than 1%, then its leftover noise.
                                    if ((contour_area / digit_area) * 100) < 1:
                                        continue

                                    # cv.imshow('digit', digit)
                                    # cv.waitKey(0)

                                    x, y, w, h = cv.boundingRect(best_cnt)
                                    top_left = (x, y)
                                    bottom_right = (x + w, y + h)
                                    digit = digit[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                                    digit = cv.morphologyEx(digit, cv.MORPH_OPEN, (3, 3))
                                    digit = utils.scale_and_centre(digit, 28, 8)

                                    sudoku_cell.cell_image = digit
                                    sudoku_cell.is_empty = False

                                    sudoku_cells[cell_name] = sudoku_cell

                        row = []

            digits_to_recognize = [cell.cell_image for cell in sudoku_cells.values() if (cell.is_empty is False)]

            if len(digits_to_recognize) > 0:
                input = np.array(digits_to_recognize)
                th, input = cv.threshold(input, 50, 255, cv.THRESH_BINARY)
                input = np.expand_dims(input, axis=3) / 255
                prediction = model.predict(input)
                prediction = np.argmax(prediction, axis=1)

                predictions = prediction.tolist()
                for cell in sudoku_cells.values():
                    if cell.is_empty is False:
                        cell.cell_value = str(predictions.pop(0))

                sudoku_string = "".join([cell.cell_value for cell in sudoku_cells.values()])

                try:
                    solved_sudoku = sudoku_solver.solve(sudoku_string)
                    for cell in solved_sudoku:
                        sudoku_cells[cell].cell_value = solved_sudoku[cell]
                except:
                    pass

            # for cell in solved_sudoku:
            #     sudoku_cells[cell].cell_value = solved_sudoku[cell]

        if solved_sudoku:
            number_image = np.zeros(image.shape + (3,), dtype=np.uint8)
            for cell in sudoku_cells.values():
                if cell.is_empty is True:
                    top_left_corner = cell.cell_corners[0]
                    bottom_right_corner = cell.cell_corners[2]
                    origin = (math.floor((top_left_corner[0] + bottom_right_corner[0]) / 2) - 14,
                              math.floor((top_left_corner[1] + bottom_right_corner[1]) / 2) + 14)

                    number_image = cv.putText(number_image, str(cell.cell_value), origin, cv.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 255, 0),
                                              2, cv.LINE_AA)

            number_image = cv.resize(number_image, (original_sudoku_size[1], original_sudoku_size[0]))
            image_points = np.array(
                [(0, 0), (original_sudoku_size[1], 0), (original_sudoku_size[1], original_sudoku_size[0]),
                 (0, original_sudoku_size[0])], dtype="float32")
            new_m = cv.getPerspectiveTransform(image_points, original_sudoku_corners)
            number_image = cv.warpPerspective(number_image, new_m, (original_image.shape[1], original_image.shape[0]))

            number_image_gray = cv.cvtColor(number_image, cv.COLOR_BGR2GRAY)
            _, number_image_mask = cv.threshold(number_image_gray, 50, 255, cv.THRESH_BINARY_INV)
            number_image_mask_inverse = cv.bitwise_not(number_image_mask)

            original_image = cv.bitwise_and(original_image, original_image, mask=number_image_mask)
            number_image = cv.bitwise_and(number_image, number_image, mask=number_image_mask_inverse)
            original_image = cv.add(original_image, number_image)

        cv.imshow('Final', original_image)

    else:
        success = video_capture.grab()
        counter += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break