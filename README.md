🧠 Fake News Detection using BERT

A Deep Learning based Fake News Detection System built using Fine-Tuned BERT and deployed using Streamlit.

Features

* Detects Fake and Real News Articles
* Fine-Tuned BERT Model
* Confidence Score Visualization
* Modern Streamlit Interface
* Real-time Prediction

Tech Stack

* Python
* PyTorch
* Transformers (Hugging Face)
* BERT
* Streamlit
* Pandas
* Scikit-Learn

Dataset

The project uses:

* Fake.csv
* True.csv

News articles are cleaned, labeled, and merged before training.

Model Performance

Metric	Score
Accuracy	99.91%
Precision	99.90%
Recall	99.90%
F1 Score	99.90%

Project Structure

fake-news-detection/

├── dataset/

├── models/

├── training/

├── app.py

├── requirements.txt

├── README.md

Installation

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt

Run Application

streamlit run app.py

Author

Devashish Gorai

Electronics & Communication Engineering (ECE)

Institute of Engineering & Management (IEM), Kolkata