# detect_grid.py

import cv2
from preprocessor import preprocess_image_for_grid, find_largest_contour, warp_image

def detect_sudoku_grid(frame):
    processed = preprocess_image_for_grid(frame)
    grid_contour = find_largest_contour(processed)
    if grid_contour is not None:
        warped, matrix = warp_image(frame, grid_contour)
        return warped, matrix
    else:
        return None, None