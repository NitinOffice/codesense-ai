import torch
import torch.nn as nn
import numpy as np

print("=== Part 1: Tensors ===")

# Create tensors
scalar = torch.tensor(5.0)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(f"Scalar: {scalar}")
print(f"Scalar shape: {scalar.shape}")

print(f"\nVector: {vector}")
print(f"Vector shape: {vector.shape}")

print(f"\nMatrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape}")

print("\n=== Part 2: Tensor Operations ===")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a mean = {a.mean()}")
print(f"a sum = {a.sum()}")

print("\n=== Part 3: Converting numpy ↔ torch ===")

numpy_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

print(f"Numpy array: {numpy_array}")
print(f"As torch tensor: {torch_tensor}")
print(f"Back to numpy: {back_to_numpy}")

print("\n=== Part 4: Build your first neural network ===")

class SimpleCodeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    

# Create the network
INPUT_SIZE = 50    # TF-IDF features
HIDDEN_SIZE = 32   # neurons in hidden layer (we choose)
OUTPUT_SIZE = 2    # good or bad

model = SimpleCodeClassifier(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
print(f"Network created!")
print(f"\nNetwork architecture:")
print(model)

print("\n=== Part 5: Count parameters ===")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total learnable parameters: {total_params}")

print("\n=== Part 6: Pass data through network ===")
# Create fake input — 1 code snippet with 50 TF-IDF features
fake_input = torch.randn(1, 50)
print(f"Input shape: {fake_input.shape}")

output = model(fake_input)
print(f"Output shape: {output.shape}")
print(f"Raw output: {output}")
print(f"(These are raw scores, not probabilities yet)")

print("\n=== Part 7: Loss function and optimizer ===")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Fake training step to see how it works
fake_label = torch.tensor([1])  # 1 = good code

# Forward pass
output = model(fake_input)
loss = criterion(output, fake_label)
print(f"Loss before training: {loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"✅ One training step complete!")
print(f"Weights have been adjusted slightly")

print("\n=== Summary ===")
print("✅ Tensors - PyTorch's version of arrays")
print("✅ Neural Network - layers of connected neurons")
print("✅ Forward pass - data flows through network")
print("✅ Loss function - measures how wrong we are")
print("✅ Backpropagation - finds what caused the error")
print("✅ Optimizer - adjusts weights to fix the error")
print("\nTomorrow: Train this network on your real code dataset!")