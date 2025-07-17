import sys
import os
sys.path.append(os.path.abspath('.'))

import cv2
from detect_grid import detect_sudoku_grid
from scripts.extract_digits import extract_digits_from_grid

cap = cv2.VideoCapture(0)
print("📸 Press 'c' to capture the Sudoku board once it's detected.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped, display = detect_sudoku_grid(frame)

    if warped is not None:
        cv2.imshow("Warped Grid", warped)

    cv2.imshow("Webcam", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c') and warped is not None:
        try:
            board = extract_digits_from_grid(warped)
            print("\n🧠 Recognized Sudoku Board:")
            for row in board:
                print(row)
        except Exception as e:
            print("❌ Error during digit extraction:", e)

cap.release()
cv2.destroyAllWindows()
