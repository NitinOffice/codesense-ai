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
""",
"""
def celsius_to_fahrenheit(celsius):
    # Convert celsius temperature to fahrenheit
    if not isinstance(celsius, (int, float)):
        raise TypeError("Temperature must be a number")
    fahrenheit = (celsius * 9/5) + 32
    return round(fahrenheit, 2)
""",
"""
def remove_duplicates(items):
    # Remove duplicate items while preserving order
    if not items:
        return []
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items
""",
"""
def validate_email(email):
    # Check if email address is valid format
    if not email or not isinstance(email, str):
        return False
    has_at = "@" in email
    has_dot = "." in email
    return has_at and has_dot
""",
"""
def calculate_average(numbers):
    # Calculate average of a list of numbers
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    total = sum(numbers)
    average = total / len(numbers)
    return round(average, 2)
""",
"""
def reverse_string(text):
    # Reverse the characters in a string
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return text[::-1]
""",
"""
def is_prime(number):
    # Check if a number is prime
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True
""",
"""
def flatten_list(nested_list):
    # Flatten a nested list into a single list
    if not nested_list:
        return []
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat
""",
"""
def count_occurrences(text, word):
    # Count how many times a word appears in text
    if not text or not word:
        return 0
    words = text.lower().split()
    return words.count(word.lower())
""",
"""
def get_file_extension(filename):
    # Extract file extension from filename
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()
""",
"""
def merge_dictionaries(dict1, dict2):
    # Merge two dictionaries safely
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both inputs must be dictionaries")
    merged = dict1.copy()
    merged.update(dict2)
    return merged
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
""",
"""
def c(x,y,z):
    a=x+y
    b=a*z
    return b
""",
"""
l=[1,2,3,4,5]
s=0
for i in range(len(l)):
    s=s+l[i]
print(s)
""",
"""
def get(u):
    import urllib.request
    r=urllib.request.urlopen(u)
    return r.read()
""",
"""
def p(n):
    for i in range(2,n):
        if n%i==0:
            return False
    return True
""",
"""
a=[]
def add(x):
    a.append(x)
def remove(x):
    a.remove(x)
def get():
    return a
""",
"""
def search(l,x):
    for i in range(len(l)):
        for j in range(len(l)):
            if l[i]==x:
                return i
    return -1
""",
"""
d={}
def s(k,v):
    d[k]=v
def g(k):
    return d[k]
""",
"""
def m(a,b,c,d,e,f,g,h):
    return a+b+c+d+e+f+g+h
""",
"""
x1=input()
x2=input()
x3=input()
print(int(x1)+int(x2)+int(x3))
""",
"""
def check(n):
    if n==1: return True
    if n==2: return True
    if n==3: return True
    if n==4: return False
    if n==5: return True
"""
]

all_code = good_code_examples + bad_code_examples
all_labels = ["good"] * len(good_code_examples) + ["bad"] * len(bad_code_examples)
print(f"Dataset ready!")
print(f"Good examples: {len(good_code_examples)}")
print(f"Bad examples: {len(bad_code_examples)}")
print(f"Total: {len(all_code)} samples")