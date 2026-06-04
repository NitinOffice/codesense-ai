# Our training dataset
# Good code = clean, has comments, handles errors, clear names
# Bad code = messy, no comments, unclear names, no error handling

good_code_examples = [ 
"""
def calculate_area(radius):
    # Calculate area of circle using formula
    pi = 3.14159
    area = pi * radius * radius
    return area
""",
"""
def get_user_age(birth_year, current_year):
    if birth_year <= 0 or current_year <= 0:
        raise ValueError("Years must be positive numbers")
    age = current_year - birth_year
    return age
""",
"""
def read_file(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
""",
"""
def find_maximum(numbers):
    # Find maximum value in a list
    if not numbers:
        return None
    maximum = numbers[0]
    for num in numbers:
        if num > maximum:
            maximum = num
    return maximum
""",
"""
class BankAccount:
    # Represents a bank account with basic operations
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        # Add money to account
        if amount > 0:
            self.balance += amount
            return True
        return False
"""
]

bad_code_examples = [
"""
def f(x):
    return 3.14159*x*x
""",
"""
def calc(a,b,c):
    z=a+b
    y=z*c
    x=y/a
    return x
""",
"""
password = "admin123"
api_key = "sk-abc123xyz"
def login(u,p):
    if p==password:
        return True
""",
"""
data = [1,2,3,4,5]
for i in range(len(data)):
    for j in range(len(data)):
        if data[i]==data[j]:
            print(data[i])
""",
"""
def process(d):
    x=[]
    for i in d:
        if i>0:
            x.append(i*2)
    return x
"""
]

all_code = good_code_examples + bad_code_examples
all_labels = ["good"] * len(good_code_examples) + ["bad"] * len(bad_code_examples)
print(f"Dataset ready!")
print(f"Good examples: {len(good_code_examples)}")
print(f"Bad examples: {len(bad_code_examples)}")
print(f"Total: {len(all_code)} samples")