from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.code_dataset import all_code, all_labels

# ── Step 1: Convert code to numbers ──────────────────────
print("=== Step 1: Preparing Data ===")
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code)
y = all_labels
print(f"Data ready: {X.shape[0]} examples, {X.shape[1]} features")

# ── Step 2: Split data ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"Train: {X_train.shape[0]} examples")
print(f"Test:  {X_test.shape[0]} examples")

# ── Step 3: Train Logistic Regression ────────────────────
print("\n=== Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Accuracy: {lr_accuracy * 100:.1f}%")

# ── Step 4: Train Random Forest ───────────────────────────
print("\n=== Random Forest ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {rf_accuracy * 100:.1f}%")

# ── Step 5: Compare both models ───────────────────────────
print("\n=== Model Comparison ===")
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 35)
print(f"{'Logistic Regression':<25} {lr_accuracy*100:>9.1f}%")
print(f"{'Random Forest':<25} {rf_accuracy*100:>9.1f}%")

if rf_accuracy >= lr_accuracy:
    print("\n✅ Random Forest wins! Using it in our app.")
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    print("\n✅ Logistic Regression wins! Keeping it.")
    best_model = lr_model
    best_model_name = "Logistic Regression"

# ── Step 6: Feature Importance ───────────────────────────
print("\n=== What words matter most? ===")
feature_names = vectorizer.get_feature_names_out()
importances = rf_model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

print("Top 10 most important words for predicting code quality:")
for i in range(min(10, len(feature_names))):
    word = feature_names[indices[i]]
    importance = importances[indices[i]]
    bar = "█" * int(importance * 200)
    print(f"  {word:<20} {importance:.4f}  {bar}")


# ── Step 7: Test on new code ──────────────────────────────
print("\n=== Testing on New Code ===")

test_cases = [
    {
        "name": "Well written function",
        "code": """
def calculate_bmi(weight_kg, height_m):
    # Calculate Body Mass Index
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)
"""
    },
    {
        "name": "Poorly written function",
        "code": """
def x(a,b):
    return a/b
"""
    },
    {
        "name": "Medium quality code",
        "code": """
def add_numbers(a, b):
    return a + b
"""
    }
]

for test in test_cases:
    # Convert to numbers
    code_numbers = vectorizer.transform([test["code"]])
    
    # Get prediction
    prediction = best_model.predict(code_numbers)[0]
    
    # Get probabilities
    probabilities = best_model.predict_proba(code_numbers)[0]
    classes = best_model.classes_
    
    # Find confidence
    predicted_index = list(classes).index(prediction)
    confidence = probabilities[predicted_index] * 100
    
    print(f"\n📝 {test['name']}")
    print(f"   Prediction: {str(prediction).upper()}")
    print(f"   Confidence: {confidence:.1f}%")