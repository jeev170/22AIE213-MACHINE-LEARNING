**Course Code:** 22AIE213 – Machine Learning  
**Lab:** 03  
**Deadline:** 01/02/2026  

---

## Requirements

- Python 3.x  
- NumPy  
- OpenCV (cv2)  
- Matplotlib  
- SciPy  
- scikit-learn  

Install the required libraries (if not already installed):

```
pip install numpy opencv-python matplotlib scipy scikit-learn
````

---

## Dataset

The dataset used in this experiment consists of **camera-captured handwritten images**.

Dataset location:

```
Lab3_Dataset/
```

Dataset structure:

```
Lab3_Dataset/
 ├── Person_1/
 │    ├── img1.jpg
 │    ├── .......
 │    └── img5.jpg
 ├── Person_2/
 ├── Person_3/
 ├── Person_4/
 └── Person_5/
```

* Each folder represents **one individual (writer)**.
* Each folder contains multiple **handwritten image samples** of that individual.
* Images are processed using OpenCV.
* Feature extraction is performed directly on image data.
* As per lab instructions, **only two classes (writers)** are selected for classification experiments.

Ensure the correct dataset path is specified in `main.py` before running the program.

---

## How to Run

1. Open Terminal / Command Prompt
2. Navigate to the project directory:

```
cd path/to/ML_LAB3
```

3. Run the main program:

```
python main.py
```

All results, metrics, and plots will be displayed sequentially.

---

## Notes

* The dataset consists **only of image files**, not spreadsheets or CSV files.
* All feature extraction is performed directly on images using OpenCV.
* All functions are defined in `lab3.py`.
* Print statements are used only in `main.py`.
* Confusion matrices are displayed as images.
* The kNN classifier demonstrates **regular fitting behavior** based on experimental results.

---

