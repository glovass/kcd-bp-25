import random
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import re

# === Log Generator ===
def generate_normal_logs(n=200):
    messages = [
        "INFO Statement: A",
        "INFO Statement: B",
        "INFO User login successful",
        "INFO File uploaded",
        "INFO Connection established"
    ]
    return [f"{random.choice(messages)}" for _ in range(n)]

def generate_anomaly_log():
    return "ERROR Statement: error"

# === Feature Extractor ===
def extract_features(logs):
    features = []
    for log in logs:
        log_level = 0  # 0=INFO, 1=ERROR, 2=WARNING, 3=CRITICAL
        if "ERROR" in log:
            log_level = 1
        elif "WARNING" in log:
            log_level = 2
        elif "CRITICAL" in log:
            log_level = 3

        has_error_keyword = int("error" in log.lower())
        has_success = int("success" in log.lower())
        features.append([log_level, has_error_keyword, has_success])
    return np.array(features)

# === Model Trainer ===
def train_model(normal_logs, model_path='svm_model.pkl', scaler_path='scaler.pkl'):
    X = extract_features(normal_logs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(gamma='auto', nu=0.01)
    model.fit(X_scaled)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Model and scaler saved.")

# === Tester ===
def test_model(normal_log, anomaly_log, model_path='svm_model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    logs = [normal_log, anomaly_log]
    X_test = extract_features(logs)
    X_test_scaled = scaler.transform(X_test)
    print(X_test_scaled)
    preds = model.predict(X_test_scaled)

    for log, pred in zip(logs, preds):
        status = "Normal ✅" if pred == 1 else "Anomaly 🚨"
        print(f"{log} => {status}")

# === Main ===
if __name__ == "__main__":
    normal_logs = generate_normal_logs()
    normal_sample = normal_logs[0]
    anomaly_sample = generate_anomaly_log()

    train_model(normal_logs)
    test_model(normal_sample, anomaly_sample)
