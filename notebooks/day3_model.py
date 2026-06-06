# Import the tools we need
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys, os
# Tell Python where to find our files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.code_dataset import all_code, all_labels

print("=== Step 1: Loading Data ===")
print(f"Total examples: {len(all_code)}")
print(f"Labels: {all_labels}")

print("\n=== Step 2: Converting Code to Numbers (TF-IDF) ===")
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code)
y = all_labels
print(f"X shape: {X.shape}  → {X.shape[0]} code snippets, {X.shape[1]} features each")
print(f"y values: {y}")

print("\n=== Step 3: Splitting into Train and Test ===")
X_train, X_test, y_train, y_test = train_test_split(
    X,          # our input numbers
    y,          # our correct labels
    test_size=0.2,      # 20% goes to test set
    random_state=42     # makes split same every time you run
)

print(f"Training examples: {X_train.shape[0]}")
print(f"Testing examples: {X_test.shape[0]}")
print(f"Training labels: {y_train}")
print(f"Testing labels: {y_test}")

print("\n=== Step 4: Training the Model ===")
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

print("\n=== Step 5: Testing the Model ===")
y_predicted = model.predict(X_test)

print(f"Correct answers:    {list(y_test)}")
print(f"Model predictions:  {list(y_predicted)}")

print("\n=== Step 6: Measuring Accuracy ===")
accuracy = accuracy_score(y_test, y_predicted)
print(f"Accuracy: {accuracy * 100:.1f}%")

print("\n=== Step 7: Detailed Report ===")
print(classification_report(y_test, y_predicted))

print("\n=== Step 8: Predict on Brand New Code ===")
new_good_code = """
def calculate_total(prices):
    # Calculate sum of all prices
    if not prices:
        return 0
    total = sum(prices)
    return total
"""

new_bad_code = """
def x(a,b):
    return a+b+c
"""

# Convert new code to numbers using the SAME vectorizer
# Important: we use transform() not fit_transform()
# Because we don't want to relearn vocabulary, just convert
new_good_numbers = vectorizer.transform([new_good_code])
new_bad_numbers = vectorizer.transform([new_bad_code])

good_prediction = model.predict(new_good_numbers)
bad_prediction = model.predict(new_bad_numbers)

print(f"New good code predicted as: {good_prediction[0]}")
print(f"New bad code predicted as:  {bad_prediction[0]}")

# Also get probability scores
good_proba = model.predict_proba(new_good_numbers)
bad_proba = model.predict_proba(new_bad_numbers)

print(f"\nGood code confidence: {good_proba[0]}")
print(f"Bad code confidence:  {bad_proba[0]}")
print(f"Classes order: {model.classes_}")