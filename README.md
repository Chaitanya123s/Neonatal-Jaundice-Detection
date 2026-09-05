# Neonatal Jaundice Detection

A machine learning based web application for neonatal jaundice screening using image processing and computer vision.

The application allows users to upload an infant image, extracts visual features using Histogram of Oriented Gradients (HOG), and uses a trained machine learning model to classify the image as Healthy Baby or Prenatal Jaundice Detected.

> Note: This project is an academic/research prototype and is not intended to replace professional medical diagnosis, bilirubin testing, or advice from a qualified healthcare professional.

## Project Overview

Neonatal jaundice is a common condition in newborns that causes yellowing of the skin and eyes due to elevated bilirubin levels.

This project explores an image-based approach for jaundice screening using computer vision and machine learning. The system processes an uploaded image, extracts HOG features, and passes those features to a trained classification model to generate a prediction.

## Workflow

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
Resize Image to 128 x 128
     |
     v
Convert to Grayscale
     |
     v
HOG Feature Extraction
     |
     v
Trained Machine Learning Model
     |
     +----------------------+
     |                      |
     v                      v
Healthy Baby        Jaundice Detected
     |                      |
     +----------+-----------+
                |
                v
        Results and Metrics
                |
                v
           PDF Report
```

## Features

* Upload infant images through the web interface
* Image preprocessing using OpenCV
* HOG based feature extraction
* Machine learning based image classification
* Healthy Baby and Jaundice Detected classification
* Accuracy, precision, recall, and F1-score calculation
* Confusion matrix visualization
* PDF report generation
* Flask based web application

## Technologies Used

| Technology   | Purpose                         |
| ------------ | ------------------------------- |
| Python       | Core programming language       |
| Flask        | Web application framework       |
| OpenCV       | Image processing                |
| NumPy        | Numerical operations            |
| Scikit-image | HOG feature extraction          |
| Scikit-learn | Machine learning and evaluation |
| Joblib       | Loading the trained model       |
| Matplotlib   | Data visualization              |
| Seaborn      | Confusion matrix visualization  |
| FPDF         | PDF report generation           |
| HTML/CSS     | Web interface                   |

## Machine Learning Approach

### Image Preprocessing

The uploaded image is resized to 128 x 128 pixels and converted to grayscale before feature extraction.

```python
img = cv2.resize(img, (128, 128))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### HOG Feature Extraction

Histogram of Oriented Gradients (HOG) is used to extract image features.

```python
features, _ = hog(
    gray,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)
```

The extracted features are then passed to the trained machine learning model.

### Prediction

The trained model is loaded using Joblib.

```python
model = joblib.load('model/jaundice_model.pkl')
```

The model produces one of the following classifications:

```text
0 - Healthy Baby
1 - Prenatal Jaundice Detected
```

## Model Evaluation

The application includes the following evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics are displayed through the web application to provide information about the model's prediction performance.

Note: Some of the current metric visualizations in the application use placeholder ground-truth values. They should not be considered a rigorous evaluation of the model's real-world or clinical performance. A separate labelled test dataset should be used for proper evaluation.

## Project Structure

```text
Neonatal-Jaundice-Detection/
|
├── archive/
│   └── ...
|
├── model/
│   └── jaundice_model.pkl
|
├── static/
│   └── ...
|
├── templates/
│   └── ...
|
├── app.py
├── app.ipynb
├── train_model.ipynb
├── dataset.lnk
├── extrainfo.txt
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/Chaitanya123s/Neonatal-Jaundice-Detection.git
```

### Navigate to the Project

```bash
cd Neonatal-Jaundice-Detection
```

### Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install flask opencv-python numpy scikit-image scikit-learn joblib matplotlib seaborn fpdf geocoder werkzeug
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000/
```

## Application Usage

1. Open the web application.
2. Upload an infant image.
3. Submit the image for processing.
4. The image is preprocessed and HOG features are extracted.
5. The trained model generates a prediction.
6. The prediction and evaluation information are displayed.
7. A PDF report can be generated from the result.


Recommended screenshots:

* Home page
* Image upload and prediction result
* Model evaluation or confusion matrix

## Project Objectives

* Develop an image-based approach for neonatal jaundice screening.
* Apply computer vision techniques for image feature extraction.
* Build a machine learning classification pipeline.
* Develop a web interface for model inference.
* Provide prediction and model evaluation information.
* Explore image-based approaches for non-invasive screening.

## Future Improvements

* Train the model using a larger and more diverse dataset.
* Compare different machine learning algorithms.
* Explore CNN and transfer learning based approaches.
* Improve image preprocessing and feature extraction.
* Perform hyperparameter optimization.
* Use proper train, validation, and test datasets.
* Perform cross-validation.
* Improve robustness under different lighting conditions.
* Evaluate the model using an independent test dataset.
* Deploy the application using a production-ready server.

## Limitations

The project is currently an academic/research prototype.

Image-based jaundice detection can be affected by factors such as:

* Lighting conditions
* Camera quality
* Image resolution
* Background conditions
* Image angle
* Dataset size and diversity

The system should not be used as a standalone medical diagnostic tool.

## Author

Chaitanya Suryawanshi

GitHub: https://github.com/Chaitanya123s

## License

This project is intended for educational and research purposes.
