from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys, os

# Tell Python where to find our files
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data.code_dataset import all_code, all_labels

# ── Train the model when the app starts ──────────────────
# Step 1: Convert code to numbers
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code)
y = all_labels

# Step 2: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
# Note: we train on ALL data here (no split) because we want
# the model to learn from every example we have
print("✅ CodeSense AI model trained and ready!")

def analyze_code(code: str) -> dict:
    """Basic analysis - counts lines, comments etc."""
    lines = code.split("\n")
    result = {
        "total_lines": len(lines),
        "empty_lines": sum(1 for line in lines if line.strip() == ""),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
    }
    return result

def predict_code_quality(code: str) -> dict:
    """
    Uses ML model to predict if code is good or bad.
    Returns prediction and confidence score.
    """
    # Convert the new code to numbers
    code_numbers = vectorizer.transform([code])

    # Get prediction (good or bad)
    prediction = model.predict(code_numbers)[0]

    # Get confidence probabilities
    probabilities = model.predict_proba(code_numbers)[0]
    classes = model.classes_

    # Find confidence for the predicted class
    predicted_index = list(classes).index(prediction)
    confidence = probabilities[predicted_index]

    return {
        "prediction": str(prediction),
        "confidence": round(float(confidence) * 100, 1),
        "all_probabilities": {
            str(classes[i]): round(float(probabilities[i]) * 100, 1)
            for i in range(len(classes))
        }
    }

def full_analysis(code: str) -> dict:
    """
    Combines basic analysis + ML prediction into one result.
    This is what the app will show to the user.
    """
    basic = analyze_code(code)
    ml_result = predict_code_quality(code)

    return {
        **basic,           # spreads all basic analysis keys into this dict
        **ml_result,       # spreads all ML result keys into this dict
    }

# ── Test it ──────────────────────────────────────────────
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
    print(f"Code to analyze:\n{test_code}")

    result = full_analysis(test_code)

    print("📊 Analysis Results:")
    print(f"   Total lines:    {result['total_lines']}")
    print(f"   Comment lines:  {result['comment_lines']}")
    print(f"   Has functions:  {result['has_functions']}")
    print(f"   Empty lines:    {result['empty_lines']}")
    print(f"\n🤖 ML Prediction:")
    print(f"   Quality:        {result['prediction'].upper()}")
    print(f"   Confidence:     {result['confidence']}%")
    print(f"   Probabilities:  {result['all_probabilities']}")