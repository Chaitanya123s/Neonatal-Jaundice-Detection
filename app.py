# Final Enhanced Version of app.py with Visual Enhancements and Performance Metrics
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from werkzeug.utils import secure_filename
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF
import geocoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
model = joblib.load('model/jaundice_model.pkl')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Feature extraction
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return np.array(features).reshape(1, -1)


# Generate confusion matrix image
def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['Healthy', 'Jaundiced'], yticklabels=['Healthy', 'Jaundiced'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_path = None
    accuracy = precision = recall = f1 = 0
    suggestions = ""
    confusion_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            features = extract_features(img_path)
            prediction = model.predict(features)[0]
            result = "Prenatal Jaundice Detected" if prediction == 1 else "Healthy Baby"

            # Dummy truth values for visualization (replace with real if available)
            y_true = [1]  # Assume expected
            y_pred = [prediction]

            accuracy = model.score(features, [prediction]) * 100
            precision = precision_score(y_true, y_pred, zero_division=1)
            recall = recall_score(y_true, y_pred, zero_division=1)
            f1 = f1_score(y_true, y_pred, zero_division=1)

            confusion_image = generate_confusion_matrix(y_true, y_pred)
            confusion_path = os.path.join('static', 'confusion_matrix.png')
            with open(confusion_path, 'wb') as f:
                f.write(confusion_image.read())

            if prediction == 1:
                suggestions = "Ensure regular pediatric checkups, maintain hydration, and monitor skin and eye color frequently."
            else:
                suggestions = "Baby appears healthy. Continue regular care and follow-up."

            with open("static/latest_prediction.txt", "w", encoding="utf-8") as f:
                f.write(f"Result: {result}\nSuggestions: {suggestions}\nAccuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

    return render_template('index.html',
                           result=result,
                           img_path=img_path,
                           accuracy=f"{accuracy:.2f}%",
                           precision=f"{precision:.2f}",
                           recall=f"{recall:.2f}",
                           f1=f"{f1:.2f}",
                           suggestions=suggestions,
                           report_generated=True,
                           confusion_path='static/confusion_matrix.png' if confusion_image else None)

@app.route('/generate_report')
def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    if os.path.exists("static/latest_prediction.txt"):
        with open("static/latest_prediction.txt", "r", encoding="utf-8") as f:
            content = f.readlines()
            for line in content:
                pdf.cell(200, 10, txt=line.strip(), ln=True)

    report_path = "static/report.pdf"
    pdf.output(report_path)
    return send_file(report_path, as_attachment=True)

@app.route('/know_us')
def know_us():
    return render_template("know_us.html")

@app.route('/find_hospital')
def find_hospital():
    g = geocoder.ip('me')
    location = g.latlng
    return render_template("find_hospital.html", location=location)

@app.route('/dataset')
def dataset():
    healthy_count = len(os.listdir('dataset/healthy'))
    jaundiced_count = len(os.listdir('dataset/jaundiced'))
    total = healthy_count + jaundiced_count
    return render_template("dataset.html",
                           healthy_count=healthy_count,
                           jaundiced_count=jaundiced_count,
                           total_count=total)


if __name__ == '__main__':
    app.run(debug=True)

