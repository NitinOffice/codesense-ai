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
""",
"""
def send_email(recipient, subject, body):
    # Validate all inputs before sending
    if not recipient or not subject:
        raise ValueError("Recipient and subject are required")
    if "@" not in recipient:
        raise ValueError("Invalid email address")
    print(f"Sending email to {recipient}")
    return True
""",
"""
def count_words(text):
    # Count number of words in a text string
    if not text or not text.strip():
        return 0
    words = text.strip().split()
    return len(words)
""",
"""
def is_palindrome(word):
    # Check if a word reads same forwards and backwards
    if not word:
        return False
    cleaned = word.lower().strip()
    return cleaned == cleaned[::-1]
""",
"""
def get_even_numbers(numbers):
    # Filter and return only even numbers from list
    if not numbers:
        return []
    even_numbers = [num for num in numbers if num % 2 == 0]
    return even_numbers
""",
"""
def safe_divide(numerator, denominator):
    # Safely divide two numbers with zero check
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    result = numerator / denominator
    return result
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
""",
"""
def fn(a,b,c,d,e):
    return a*b+c-d/e
""",
"""
x=input()
y=input()
print(int(x)+int(y))
""",
"""
import os
def d(p):
    os.remove(p)
    return 1
""",
"""
def check(l):
    r=[]
    for i in range(len(l)):
        if l[i]%2==0:
            r.append(l[i])
    return r
""",
"""
global_data = []
def add(x):
    global global_data
    global_data.append(x)
    return global_data
"""
]

all_code = good_code_examples + bad_code_examples
all_labels = ["good"] * len(good_code_examples) + ["bad"] * len(bad_code_examples)
print(f"Dataset ready!")
print(f"Good examples: {len(good_code_examples)}")
print(f"Bad examples: {len(bad_code_examples)}")
print(f"Total: {len(all_code)} samples")