import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from data.code_dataset import all_code, all_labels

vectorizer = TfidfVectorizer(
    max_features=50,
    lowercase=True,
)
X = vectorizer.fit_transform(all_code)

print(f"Original code: {len(all_code)} snippets (just text)")
print(f"After TF-IDF: {X.shape} matrix")
print(f"  → {X.shape[0]} code snippets")
print(f"  → {X.shape[1]} unique words/tokens found")

words = vectorizer.get_feature_names_out()
print(f"\nTop words TF-IDF found in your code:")
print(list(words[:20]))

print(f"\nFirst code snippet as numbers (first 10 values):")
print(X[0].toarray()[0][:10].round(3))