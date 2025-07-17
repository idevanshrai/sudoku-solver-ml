# preprocessor.py

import cv2
import numpy as np

def preprocess_image_for_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def find_largest_contour(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                best_cnt = approx
                max_area = area
    return best_cnt

def reorder_points(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    new_pts[0] = pts[np.argmin(s)]
    new_pts[2] = pts[np.argmax(s)]
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[3] = pts[np.argmax(diff)]
    return new_pts

def warp_image(img, pts):
    reordered = reorder_points(pts)
    (tl, tr, br, bl) = reordered
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    side = max(maxWidth, maxHeight)
    if side == 0: return None, None
    dst_pts = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(reordered, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (side, side))
    return warped, matrix

def extract_and_preprocess_digit(cell_img):
    if cell_img is None: return None
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h, w = thresh.shape
    border_size = int(min(h, w) * 0.1)
    thresh = thresh[border_size:h-border_size, border_size:w-border_size]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 5 or h < 5: return None
    digit_roi = thresh[y:y+h, x:x+w]
    side = max(w, h)
    padded_digit = np.zeros((side, side), dtype=np.uint8)
    dx = (side - w) // 2
    dy = (side - h) // 2
    padded_digit[dy:dy+h, dx:dx+w] = digit_roi
    final_digit = cv2.resize(padded_digit, (28, 28))
    final_digit = final_digit.astype("float32") / 255.0
    final_digit = np.expand_dims(final_digit, axis=-1)
    return final_digit