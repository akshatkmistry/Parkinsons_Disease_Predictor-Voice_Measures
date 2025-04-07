# Parkinson's Disease Predictor - Voice Measures

🎙️ Predict Parkinson's disease from voice measures using a Random Forest Classifier, built with scikit-learn and deployed as a Streamlit web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parkinsons-disease-predictor-voicemeasures.streamlit.app/) 

---

## 🧠 Overview

This project utilizes a Random Forest Classifier to detect Parkinson’s disease based on vocal biometrics from the [UCI Parkinson's dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons). With a high accuracy of **94.87%**, the app features EDA visualizations, model training, and a user-friendly prediction interface.

 
- **Author**: [Akshat Mistry](https://github.com/akshatkmistry)

---

## 🚀 Features

- **Model**: Random Forest (100 estimators)
- **Accuracy**: 94.87% on test data
- **Input Features**:
  - MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, PPE
- **Analysis**:
  - Correlation heatmap, boxplots, feature importance
- **Streamlit Web App**:
  - Manual input fields
  - CSV upload for batch predictions

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akshatkmistry/Parkinsons_Disease_Predictor-Voice_Measures.git
   cd Parkinsons_Disease_Predictor-Voice_Measures
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # or source venv/bin/activate (Linux/Mac)
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run src/app.py
   ```
---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **File**: `parkinsons.csv`
- **Samples**: 195  
  - Healthy (0): 48  
  - Parkinson’s (1): 147  
- **Target**: `status` (0 = Healthy, 1 = Parkinson’s)

---

## 🧠 Model Chosen

- **Algorithm**: Random Forest Classifier  
- **Why Random Forest?**
  - Handles both numerical and categorical features well
  - Robust to outliers and overfitting
  - Provides high accuracy with minimal hyperparameter tuning

---

## 📊 Performance Metrics

- **Accuracy**: ~94% on test set
- **Train-Test Split**: 80-20
- **Model Evaluation**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)


---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m "Add feature"`
4. Push to GitHub: `git push origin feature-name`
5. Open a Pull Request

---

## 📜 License

MIT License © Akshat Mistry, 2025


###### *Made with ❤️ for health-tech innovation.*

