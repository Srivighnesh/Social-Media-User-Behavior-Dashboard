# 📊 Social Media User Behavior Analysis Dashboard

A Streamlit dashboard for analyzing social media user behavior using **Exploratory Data Analysis (EDA)** and **Unsupervised Machine Learning** techniques. The project applies **K-Means** and **DBSCAN** clustering to identify user behavior patterns and provides interactive visualizations for cluster profiling.

**Projec URL: ** https://dasboard-socaial-media.streamlit.app/

---
## 🚀 Features

- Interactive Streamlit Dashboard
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- K-Means Clustering
- DBSCAN Clustering
- Cluster Evaluation
  - Silhouette Score
  - Davies-Bouldin Index
- Cluster Profiling
- Numerical Feature Pattern Analysis
- Categorical Feature Pattern Analysis
- Interactive Plotly Visualizations

---

## 📂 Dashboard Modules

### 📈 Feature Overview
- Dataset summary
- Feature distributions
- Statistical overview

### 🎯 K-Means Clustering
- Automatic clustering
- Cluster visualization
- Cluster profiling
- PCA visualization

### 🔍 DBSCAN Clustering
- Density-based clustering
- Noise detection
- Cluster visualization

### 📊 Algorithm Comparison
- Compare K-Means and DBSCAN
- Silhouette Score
- Davies-Bouldin Index

### 📌 Pattern Comparison
- Numerical feature comparison across clusters
- Categorical feature comparison across clusters

---

## 📊 Categories Analyzed

The dashboard analyzes user behavior in the following categories:

- User Behavior
- Engagement
- Social Influence
- Spending & Marketing
- Demographic & Lifestyle
- Mental Health & Usage Impact
- Platform Usage Behavior

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Matplotlib

---

## 📁 Project Structure

```
Social-Media-User-Behavior-Dashboard/
│
├── app.py
├── data/
│   └── social_media_dataset.csv
├── utils/
├── images/
├── requirements.txt
├── README.md
└── assets/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Social-Media-User-Behavior-Dashboard.git
```

Move into the project folder

```bash
cd Social-Media-User-Behavior-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Machine Learning Techniques

### K-Means
- Cluster users based on behavioral patterns
- Visualize clusters using PCA
- Profile each cluster

### DBSCAN
- Detect density-based clusters
- Identify outliers
- Compare with K-Means

---

## 📈 Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Index

These metrics help evaluate the quality of the generated clusters.

---

## 📷 Dashboard Preview

You can add screenshots of:

- Home Page
- K-Means Results
- DBSCAN Results
- Pattern Comparison
- Algorithm Comparison

Example:

```
images/
├── dashboard.png
├── kmeans.png
├── dbscan.png
```

---

## 🎯 Future Improvements

- Hierarchical Clustering
- Gaussian Mixture Models (GMM)
- Recommendation System
- User Segmentation Export
- Download Reports
- Interactive Filtering
- Cluster Prediction for New Users

---

## 📄 License

This project is created for educational and research purposes.
