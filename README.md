# Credit_card-fraud-model



---

```md
# Credit Card Fraud Detection

## Description
This project applies **machine learning** techniques to detect fraudulent transactions in a credit card dataset. It uses **anomaly detection** (`Isolation Forest`, `Local Outlier Factor`) and a **classification model** (`Logistic Regression`) to identify fraudulent transactions.

---

## 📂 Project Structure
```
credit-card-fraud-detection/
│── data/
│   ├── creditcard.csv         # Dataset used for training and evaluation
│── notebooks/
│   ├── fraud_detection.ipynb  # Jupyter Notebook implementation (optional)
│── src/
│   ├── fraud_detection.py     # Main Python script
│── reports/
│   ├── output.txt             # Model results and findings
│── README.md                  # Project documentation
│── requirements.txt           # Dependencies and libraries
│── .gitignore                 # Files to ignore in Git
```

---

## 🔧 Installation & Setup
1. **Clone the repository:**  
   
   cd credit-card-fraud-detection
   ```
2. **Create a virtual environment (optional but recommended):**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Project Overview
This project detects fraud in credit card transactions using **unsupervised and supervised learning** techniques.

### ✅ Key Features:
- **Data Preprocessing** – Standardization and feature engineering  
- **Anomaly Detection** – `Isolation Forest` and `Local Outlier Factor (LOF)`  
- **Supervised Learning Model** – `Logistic Regression`  
- **Performance Metrics** – Accuracy, Precision, Recall, Confusion Matrix  

---

## 📊 Results Summary

| Model                  | Accuracy | ROC-AUC Score |
|------------------------|----------|--------------|
| **Logistic Regression** | 99.92%   | 0.956        |
| **Decision Tree**       | 99.91%   | 0.877        |
| **Neural Network**      | 99.94%   | 0.979        |

**Key Findings:**  
- `Isolation Forest` and `LOF` detected **2,849 potential fraud cases**.  
- **Neural Networks** achieved the highest AUC score (**0.979**), making it the most effective classifier.  
- **Precision-Recall Tradeoff**: The model successfully detects fraud but has some false positives.

---

## 🛠️ Usage
To run the fraud detection script, execute:
```bash
python src/fraud_detection.py
```

---



---

## 👨‍💻 Contributors
- [Abraham Olatunde] 
