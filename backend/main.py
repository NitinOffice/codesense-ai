# backend/main.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os
import re

# ── Load saved models ─────────────────────────────────────
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

def load_models():
    """
    Load trained models from disk.
    Much faster than retraining every time.
    """
    vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    lr_model = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    model_info = joblib.load(os.path.join(models_dir, "model_info.pkl"))
    return vectorizer, rf_model, lr_model, model_info


# Load models when app starts
try:
    vectorizer, rf_model, lr_model, model_info = load_models()
    print("✅ Models loaded from disk instantly!")
    print(f"   Random Forest accuracy: {model_info['random_forest_accuracy']}%")
    print(f"   Trained on: {model_info['training_examples']} examples")
except FileNotFoundError:
    print("❌ Models not found! Run: python backend/train_model.py first")
    exit(1)


def analyze_code(code: str) -> dict:
    """Basic analysis — counts lines, detects issues"""
    lines = code.split("\n")
    issues = []

    if not any(line.strip().startswith("#") for line in lines):
        issues.append("No comments found")

    if "try" not in code and "except" not in code:
        issues.append("No error handling (try/except)")

    single_letters = re.findall(r'\b[a-z]\b', code)
    if len(single_letters) > 2:
        issues.append(f"Too many single-letter variables: {set(single_letters)}")

    if len([l for l in lines if len(l) > 79]) > 0:
        issues.append("Some lines exceed 79 characters (PEP8 standard)")

    return {
        "total_lines": len(lines),
        "empty_lines": sum(1 for line in lines if line.strip() == ""),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
        "has_error_handling": "try" in code and "except" in code,
        "issues_found": issues,
        "issues_count": len(issues)
    }


def predict_code_quality(code: str) -> dict:
    """Predict code quality using Random Forest"""
    code_numbers = vectorizer.transform([code])
    prediction = str(rf_model.predict(code_numbers)[0])
    probabilities = rf_model.predict_proba(code_numbers)[0]
    classes = rf_model.classes_
    predicted_index = list(classes).index(prediction)
    confidence = round(float(probabilities[predicted_index]) * 100, 1)

    # Top 5 most important features
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [str(feature_names[indices[i]]) for i in range(min(5, len(feature_names)))]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "all_probabilities": {
            str(classes[i]): round(float(probabilities[i]) * 100, 1)
            for i in range(len(classes))
        },
        "top_features": top_features
    }

def full_analysis(code: str) -> dict:
    """Complete analysis combining everything"""
    basic = analyze_code(code)
    ml = predict_code_quality(code)

    if ml["prediction"] == "good":
        quality_score = int(ml["confidence"])
    else:
        quality_score = int(100 - ml["confidence"])

    # Generate suggestions based on issues
    suggestions = []
    for issue in basic["issues_found"]:
        if "comment" in issue.lower():
            suggestions.append("Add comments explaining what your functions do")
        if "error" in issue.lower():
            suggestions.append("Wrap risky operations in try/except blocks")
        if "single-letter" in issue.lower():
            suggestions.append("Use descriptive variable names like 'count' instead of 'c'")
        if "79" in issue:
            suggestions.append("Break long lines using parentheses or backslash")

    return {
        **basic,
        **ml,
        "quality_score": quality_score,
        "suggestions": suggestions
    }

if __name__ == "__main__":
    test_code = """
def calculate_discount(price, discount_percent):
    # Calculate final price after discount
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    return final_price
"""
    result = full_analysis(test_code)

    print("\n=== CodeSense AI - Full Analysis ===")
    print(f"\n📊 Basic Analysis:")
    print(f"   Total lines:        {result['total_lines']}")
    print(f"   Comment lines:      {result['comment_lines']}")
    print(f"   Has error handling: {result['has_error_handling']}")
    print(f"   Issues found:       {result['issues_found']}")

    print(f"\n🤖 ML Prediction:")
    print(f"   Quality Score:      {result['quality_score']}/100")
    print(f"   Prediction:         {result['prediction'].upper()}")
    print(f"   Confidence:         {result['confidence']}%")

    print(f"\n💡 Suggestions:")
    if result['suggestions']:
        for s in result['suggestions']:
            print(f"   → {s}")
    else:
        print("   → Code looks good! No major issues found.")