from sklearn.feature_extraction.text import TfidfVectorizer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data.code_dataset import all_code, all_labels
def analyze_code(code: str) -> dict:
    lines = code.split("\n")
    result = {
        "total_lines": len(lines),
        "empty_lines": sum(1 for line in lines if line.strip() == ""),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
    }
    return result


def code_to_numbers(code: str):
    """Convert code to TF-IDF numbers so ML can read it"""
    vectorizer = TfidfVectorizer(max_features=50)
    vectorizer.fit(all_code)
    numbers = vectorizer.transform([code])
    return numbers, vectorizer


sample_code = """
def greet_user(name):
    # Say hello to the user
    if not name:
        return "Please provide a name"
    message = f"Hello, {name}!"
    return message
"""

print("=== CodeSense AI - Day 2 ===")
print("\n1. Basic Analysis:")
analysis = analyze_code(sample_code)
for key, val in analysis.items():
    print(f"   {key}: {val}")


print("\n2. Converting to numbers for ML...")
numbers, vectorizer = code_to_numbers(sample_code)
print(f"   Code converted to shape: {numbers.shape}")
print(f"   Ready for ML model ✅")
