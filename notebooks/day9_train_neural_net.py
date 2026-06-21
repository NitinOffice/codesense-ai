import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.code_dataset import all_code, all_labels

print("=== Step 1: Prepare Data ===")

# Convert code to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
X = vectorizer.fit_transform(all_code).toarray()

# Convert labels from strings to numbers
# "bad" → 0, "good" → 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(all_labels)

print(f"Labels before encoding: {all_labels[:3]}")
print(f"Labels after encoding:  {y[:3]}")
print(f"Mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training: {X_train.shape[0]} examples")
print(f"Testing:  {X_test.shape[0]} examples")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nTensor shapes:")
print(f"X_train: {X_train_tensor.shape}")
print(f"y_train: {y_train_tensor.shape}")

print("\n=== Step 2: Define Neural Network ===")

class CodeClassifierNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

model = CodeClassifierNN()
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

print("\n=== Step 3: Setup Training ===")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.5
)

EPOCHS = 150
BATCH_SIZE = 8

print("\n=== Step 4: Training Loop ===")

train_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, len(X_train_tensor), BATCH_SIZE):
        # Get batch
        X_batch = X_train_tensor[i:i+BATCH_SIZE]
        y_batch = y_train_tensor[i:i+BATCH_SIZE]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1

        # Update learning rate
        scheduler.step()
    
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Evaluate on test set every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor) * 100
            test_accuracies.append(accuracy)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Test Accuracy: {accuracy:.1f}% | "
                  f"LR: {current_lr:.6f}")


print("\n=== Step 5: Final Evaluation ===")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    # Convert back to label names
    predicted_labels = label_encoder.inverse_transform(predicted.numpy())
    actual_labels = label_encoder.inverse_transform(y_test_tensor.numpy())
    
    correct = (predicted == y_test_tensor).sum().item()
    final_accuracy = correct / len(y_test_tensor) * 100

    print(f"\nFinal Test Accuracy: {final_accuracy:.1f}%")
    print(f"\nPredictions vs Actual:")
    for pred, actual in zip(predicted_labels, actual_labels):
        status = "✅" if pred == actual else "❌"
        print(f"  {status} Predicted: {pred:<6} | Actual: {actual}")


print("\n=== Step 6: Compare with Random Forest ===")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions) * 100

print(f"Random Forest accuracy:  {rf_accuracy:.1f}%")
print(f"Neural Network accuracy: {final_accuracy:.1f}%")

if final_accuracy >= rf_accuracy:
    print("\n🧠 Neural Network wins!")
else:
    print("\n🌲 Random Forest wins (need more data for NN to shine)")
    print("   This is normal — NN needs more data than RF")
    print("   Next step: CodeBERT will beat both!")

print("\n=== Step 7: Test on brand new code ===")

def predict_with_nn(code, model, vectorizer, label_encoder):
    """Predict code quality using neural network"""
    model.eval()
    
    # Convert code to numbers
    code_vector = vectorizer.transform([code]).toarray()
    code_tensor = torch.FloatTensor(code_vector)
    
    with torch.no_grad():
        output = model(code_tensor)
        
        # Convert raw scores to probabilities
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
    
    predicted_label = label_encoder.inverse_transform(predicted_class.numpy())[0]
    confidence = probabilities[0][predicted_class].item() * 100
    
    return predicted_label, round(confidence, 1)

test_cases = [
    ("Well commented with error handling", """
def get_user_data(user_id):
    # Fetch user data from database safely
    if not user_id or not isinstance(user_id, int):
        raise ValueError("user_id must be a positive integer")
    try:
        data = database.query(f"SELECT * FROM users WHERE id={user_id}")
        return data
    except DatabaseError as e:
        print(f"Database error: {e}")
        return None
"""),
    ("Single letter vars, no comments", """
def x(a,b,c):
    z=a+b
    y=z*c
    return y
"""),
]

print("\nNeural Network Predictions:")
for name, code in test_cases:
    label, confidence = predict_with_nn(code, model, vectorizer, label_encoder)
    icon = "✅" if label == "good" else "❌"
    print(f"\n{icon} {name}")
    print(f"   Prediction: {label.upper()} ({confidence}% confident)")