# backend/train_model.py
# Run this file once to train and save the model
# After this, the app loads instantly without retraining

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data.code_dataset import all_code, all_labels

print("=== CodeSense AI - Model Training ===")
print(f"Training on {len(all_code)} examples...")

# ── Step 1: Convert code to numbers ──────────────────────
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code)
y = all_labels

# ── Step 2: Quick evaluation before saving ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate Random Forest
rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
rf_eval.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_eval.predict(X_test))

# Train and evaluate Logistic Regression
lr_eval = LogisticRegression(max_iter=1000)
lr_eval.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_eval.predict(X_test))

print(f"Random Forest accuracy:     {rf_accuracy * 100:.1f}%")
print(f"Logistic Regression accuracy:{lr_accuracy * 100:.1f}%")

# ── Step 3: Train FINAL models on ALL data ───────────────
# No train/test split here — we want to learn from everything
print("\nTraining final models on full dataset...")

final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
final_rf.fit(X, y)

final_lr = LogisticRegression(max_iter=1000)
final_lr.fit(X, y)

print("✅ Models trained on full dataset!")

# ── Step 4: Create models folder and save ────────────────
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(models_dir, exist_ok=True)

# Save vectorizer
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)
print(f"✅ Vectorizer saved → {vectorizer_path}")

# Save Random Forest
rf_path = os.path.join(models_dir, "random_forest.pkl")
joblib.dump(final_rf, rf_path)
print(f"✅ Random Forest saved → {rf_path}")

# Save Logistic Regression
lr_path = os.path.join(models_dir, "logistic_regression.pkl")
joblib.dump(final_lr, lr_path)
print(f"✅ Logistic Regression saved → {lr_path}")

# Save model info (accuracy scores etc.)
model_info = {
    "random_forest_accuracy": round(rf_accuracy * 100, 1),
    "logistic_regression_accuracy": round(lr_accuracy * 100, 1),
    "training_examples": len(all_code),
    "features": 50,
    "best_model": "random_forest" if rf_accuracy >= lr_accuracy else "logistic_regression"
}
info_path = os.path.join(models_dir, "model_info.pkl")
joblib.dump(model_info, info_path)
print(f"✅ Model info saved → {info_path}")

print("\n=== Summary ===")
print(f"Files saved in: models/")
print(f"  - vectorizer.pkl")
print(f"  - random_forest.pkl")
print(f"  - logistic_regression.pkl")
print(f"  - model_info.pkl")
print(f"\nBest model: {model_info['best_model']}")
print(f"Ready to use! Run the app now.")