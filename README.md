# Sudoku Solver using Machine Learning 🎯📷

**Sudoku Solver** is a real-time computer vision project that detects a Sudoku grid using your webcam, extracts each cell, and recognizes digits using a custom-trained Convolutional Neural Network (CNN). Built for speed, modularity, and hands-on learning.

---

## 🔧 Features

- 🎥 **Live Webcam Input** – Captures Sudoku puzzles in real time.  
- 🧠 **CNN Digit Recognition** – Predicts digits using a model trained on MNIST.  
- 🔍 **Grid Detection** – Uses contour analysis and perspective warp to isolate the board.  
- 🧪 **Preprocessing Pipeline** – Includes grayscale, blur, and adaptive thresholding.  
- 🧾 **9x9 Matrix Output** – Displays the full puzzle structure on terminal.  
- 💻 **Keyboard Interaction** – Press `'c'` to capture and detect digits.  

---

## 🛠️ Tech Stack

- **Language**: Python 3.12  
- **Computer Vision**: OpenCV  
- **Deep Learning**: Keras (TensorFlow backend)  
- **Data**: MNIST Handwritten Digits  
- **Environment**: macOS, PyCharm IDE  

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/idevanshrai/sudoku-solver-ml.git
cd sudoku-solver-ml
````

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python test_recognize_digits.py
```

---

## 📁 Project Structure

```
sudoku-solver-ml/
│
├── model/
│   └── model/
|         └── model/
│               └── digit_model.h5           # Trained CNN on MNIST
│
├── scripts/
│   ├── detect_grid.py               # Detects grid from webcam
│   └── extract_digits.py            # Extracts digits from warped image
│
├── utils/
│   └── preprocessor.py              # Helper functions for preprocessing
│
├── test_recognize_digits.py         # Main script to test grid + digit recognition
└── README.md                        # This file
```

---

## 🧠 Model Training

* The model was trained **locally** on the **MNIST dataset**, containing 70,000 handwritten digit images (60K train, 10K test).
* Training was done on **CPU**, yet completed quickly due to the simplicity of the CNN and data.
* The final model achieved **\~97% accuracy** and is saved as:

  ```
  model/model/digit_model.h5
  ```

---

## ⚠️ macOS SSL Bypass Note 

If you encounter SSL certificate errors on macOS while loading the model or fetching packages:

```bash
/Applications/Python\ 3.10/Install\ Certificates.command
```

Alternatively, you may need to set:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

**⚠️ Caution**: This bypasses certificate verification. Do **not** use in production. Only for trusted local development.

---

## 📌 Use Cases

* Real-time Sudoku digit extraction
* Digit segmentation and CNN model testing
* Computer vision experimentation
* Project showcase for ML/CV portfolio

---

## 🗺️ To-Do (Planned Features)

* [ ] Solve the Sudoku puzzle using backtracking
* [ ] UI overlay with solved digits
* [ ] Export detected board to CSV or JSON
* [ ] Add mobile phone support for live feed

---

## 🙌 Acknowledgments

* Built by [Devansh Rai](https://github.com/idevanshrai)
* Trained with [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* Inspired by OpenCV, TensorFlow/Keras tutorials

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

