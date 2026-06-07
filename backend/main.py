from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data.code_dataset import all_code, all_labels

# ── Train both models, keep the best ─────────────────────
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code)
y = all_labels

# Train Random Forest (our primary model now)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Train Logistic Regression (backup/comparison)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

print("✅ CodeSense AI models trained and ready!")
print(f"   Primary model: Random Forest (100 trees)")
print(f"   Backup model:  Logistic Regression")

def analyze_code(code: str) -> dict:
    """Basic analysis — counts lines, comments etc."""
    lines = code.split("\n")
    
    # Count issues found in code
    issues = []
    
    if not any(line.strip().startswith("#") for line in lines):
        issues.append("No comments found")
    
    if "try" not in code and "except" not in code:
        issues.append("No error handling")
    
    # Check for single letter variable names (bad practice)
    import re
    single_letters = re.findall(r'\b[a-z]\b', code)
    if len(single_letters) > 2:
        issues.append(f"Too many single-letter variables: {set(single_letters)}")
    
    return {
        "total_lines": len(lines),
        "empty_lines": sum(1 for line in lines if line.strip() == ""),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
        "has_error_handling": "try" in code and "except" in code,
        "issues_found": issues
    }

def predict_code_quality(code: str, model_type: str = "random_forest") -> dict:
    """
    Predict if code is good or bad.
    model_type: 'random_forest' or 'logistic_regression'
    """
    # Pick which model to use
    model = rf_model if model_type == "random_forest" else lr_model
    
    # Convert code to numbers
    code_numbers = vectorizer.transform([code])
    
    # Get prediction
    prediction = str(model.predict(code_numbers)[0])
    
    # Get probabilities
    probabilities = model.predict_proba(code_numbers)[0]
    classes = model.classes_
    
    # Calculate confidence
    predicted_index = list(classes).index(prediction)
    confidence = round(float(probabilities[predicted_index]) * 100, 1)
    
    # Get feature importance for this prediction (Random Forest only)
    top_features = []
    if model_type == "random_forest":
        feature_names = vectorizer.get_feature_names_out()
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [
            str(feature_names[indices[i]])
            for i in range(min(5, len(feature_names)))
        ]
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "all_probabilities": {
            str(classes[i]): round(float(probabilities[i]) * 100, 1)
            for i in range(len(classes))
        },
        "top_features": top_features,
        "model_used": model_type
    }

def full_analysis(code: str) -> dict:
    """Complete analysis — basic + ML prediction combined"""
    basic = analyze_code(code)
    ml_result = predict_code_quality(code)
    
    # Generate a quality score 0-100
    confidence = ml_result["confidence"]
    if ml_result["prediction"] == "good":
        quality_score = int(confidence)
    else:
        quality_score = int(100 - confidence)
    
    return {
        **basic,
        **ml_result,
        "quality_score": quality_score
    }

# ── Test it ───────────────────────────────────────────────
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
    print("\n=== CodeSense AI - Full Analysis ===")
    result = full_analysis(test_code)
    
    print(f"\n📊 Basic Analysis:")
    print(f"   Total lines:       {result['total_lines']}")
    print(f"   Comment lines:     {result['comment_lines']}")
    print(f"   Has functions:     {result['has_functions']}")
    print(f"   Has error handling:{result['has_error_handling']}")
    print(f"   Issues found:      {result['issues_found']}")
    
    print(f"\n🤖 ML Prediction:")
    print(f"   Quality Score:     {result['quality_score']}/100")
    print(f"   Prediction:        {result['prediction'].upper()}")
    print(f"   Confidence:        {result['confidence']}%")
    print(f"   Probabilities:     {result['all_probabilities']}")
    print(f"   Key factors:       {result['top_features']}")
    print(f"   Model used:        {result['model_used']}")