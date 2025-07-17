# app.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from detect_grid import detect_sudoku_grid
from preprocessor import extract_and_preprocess_digit
from solver import solve_sudoku, is_board_valid
import multiprocessing
import queue


def solver_worker(board, result_queue):
    if solve_sudoku(board):
        result_queue.put(board)
    else:
        result_queue.put(None)


def main():
    try:
        model = load_model('model/model/model/digit_model.h5')
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("\n[INFO] Starting Sudoku Solver...")
    print("[INFO] Press 'q' to quit.")

    grid_found_and_waiting = False
    detected_board_state = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        if not grid_found_and_waiting:
            cv2.putText(display_frame, "Status: Searching for grid...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 165, 0), 2)
            warped_grid, M = detect_sudoku_grid(frame)

            if warped_grid is not None and M is not None and np.linalg.det(M) != 0:
                board = np.zeros((9, 9), dtype=int)
                cell_side = warped_grid.shape[0] // 9

                cells_to_predict, cell_positions = [], []
                for i in range(9):
                    for j in range(9):
                        cell = warped_grid[i * cell_side:(i + 1) * cell_side, j * cell_side:(j + 1) * cell_side]
                        digit_img = extract_and_preprocess_digit(cell)
                        if digit_img is not None:
                            cells_to_predict.append(digit_img)
                            cell_positions.append((i, j))

                if cells_to_predict:
                    cells_array = np.array(cells_to_predict)
                    predictions = model.predict(cells_array, verbose=0)
                    for i, pos in enumerate(cell_positions):
                        digit = np.argmax(predictions[i])
                        confidence = np.max(predictions[i])
                        if confidence > 0.8:  # Slightly lowered confidence for better detection
                            board[pos[0], pos[1]] = digit

                detected_board_state = {"board": board, "original_board": board.copy(), "warped_grid": warped_grid,
                                        "M": M}
                print("\n--- Detected Board ---");
                print(board);
                print("----------------------")
                grid_found_and_waiting = True
        else:
            cv2.putText(display_frame, "Press 'K' to Solve or 'R' to Rescan", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            grid_found_and_waiting = False
        elif key == ord('k') and grid_found_and_waiting:
            board = detected_board_state["board"]
            original_board = detected_board_state["original_board"]
            warped_grid = detected_board_state["warped_grid"]
            M = detected_board_state["M"]
            cell_side = warped_grid.shape[0] // 9

            # ##################################################################
            # ## REFINED SOLVING LOGIC WITH CLEAR FEEDBACK                    ##
            # ##################################################################

            # First, check if the initial board is valid
            if not is_board_valid(original_board.copy()):
                print("Validation Failed: The detected board violates Sudoku rules.")
                cv2.putText(display_frame, "Status: Invalid Board! Press 'R' to rescan.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # If the board is valid, try to solve it with a timeout
                print("Board is valid. Attempting to solve...")
                result_queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=solver_worker, args=(board, result_queue))
                p.start()
                p.join(timeout=0.5)

                solved = False
                if p.is_alive():
                    p.terminate();
                    p.join()
                    print("Solver timed out.")
                else:
                    try:
                        solved_board = result_queue.get_nowait()
                        if solved_board is not None:
                            board = solved_board;
                            solved = True
                            print("Solver found a solution.")
                        else:
                            print("Solver finished, but found no solution.")
                    except queue.Empty:
                        print("Solver finished, but queue was empty.")

                # Overlay the final result
                if solved:
                    solution_overlay = np.zeros_like(warped_grid)
                    for i in range(9):
                        for j in range(9):
                            if original_board[i][j] == 0:
                                text = str(board[i][j])
                                text_x = j * cell_side + (cell_side // 2) - 15;
                                text_y = i * cell_side + (cell_side // 2) + 15
                                cv2.putText(solution_overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                            (0, 255, 0), 3)
                    M_inv = np.linalg.inv(M)
                    h, w, _ = display_frame.shape
                    unwarped_solution = cv2.warpPerspective(solution_overlay, M_inv, (w, h))
                    display_frame = cv2.addWeighted(display_frame, 1, unwarped_solution, 1, 0)
                    cv2.putText(display_frame, "Status: Solved! Press 'R' to scan again.", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Status: Unsolvable. Press 'R' to rescan.", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sudoku Solver ML", display_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()