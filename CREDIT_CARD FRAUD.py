import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import time
import dask.dataframe as dd

# Load dataset efficiently using Dask (for scalability)
df_dask = dd.read_csv(r"C:\Users\Abraham\Desktop\DOINGS\creditcard.csv", dtype={'Time': 'float64'})
df = df_dask.compute()

# Display dataset information
print("\nData Information:")
print(df.info())

print("\nData Headings:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop 'Time' column and normalize 'Amount'
df.drop(columns=['Time'], inplace=True)
scaler = StandardScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# Separating features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(X)
df['anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# Train Local Outlier Factor (LOF) for anomaly detection
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df['lof_score'] = lof.fit_predict(X)
df['lof_anomaly'] = df['lof_score'].apply(lambda x: 1 if x == -1 else 0)

# Display detected anomalies
fraud_cases = df[df['anomaly'] == 1]
lof_fraud_cases = df[df['lof_anomaly'] == 1]
print(f"Isolation Forest detected {len(fraud_cases)} potential fraud cases")
print(f"LOF detected {len(lof_fraud_cases)} potential fraud cases")

# Compare detected anomalies with actual fraud cases
true_fraud = df[df['Class'] == 1]
detected_fraud = df[(df['anomaly'] == 1) | (df['lof_anomaly'] == 1)]

# Performance metrics
true_positives = len(detected_fraud[detected_fraud['Class'] == 1])
false_positives = len(detected_fraud[detected_fraud['Class'] == 0])
false_negatives = len(true_fraud) - true_positives

print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

# Splitting data for machine learning models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Machine Learning Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, solver='adam', random_state=42)
}

# Train models
def train_model(name, model):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} ROC AUC Score: {roc_auc}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return model

trained_models = {name: train_model(name, model) for name, model in models.items()}

# Real-time Fraud Detection Simulation

def real_time_fraud_detection(model, new_data, scaler, feature_names):
    """Simulates real-time fraud detection by predicting transactions one by one."""
    
    # Convert new_data to a DataFrame with the correct feature names
    new_data_df = pd.DataFrame([new_data], columns=feature_names)

    # Ensure column order matches the trained scaler
    new_data_df = new_data_df[feature_names]

    # Convert to numpy array before scaling (avoids feature name mismatches)
    new_data_scaled = scaler.transform(new_data_df.to_numpy())

    # Predict fraud status
    prediction = model.predict(new_data_scaled)[0]

    if prediction == 1:
        print("⚠️ Fraudulent Transaction Detected! ⚠️")
    else:
        print("✅ Transaction is Normal.")

