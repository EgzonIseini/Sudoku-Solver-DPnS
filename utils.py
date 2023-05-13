import numpy as np
import math
import operator
import cv2

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

def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))