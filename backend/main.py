# CodeSense AI - Day 1
# This is the starting point of our project

def analyze_code(code: str) -> dict:
    """
    Takes code as input and returns basic info about it.
    Like a first glance before the AI does deep analysis.
    """
    lines = code.split("\n")
    
    result = {
        "total_lines": len(lines),
        "empty_lines": sum(1 for line in lines if line.strip() == ""),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
    }
    
    return result


# Test it
sample_code = """
# This is a sample function
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 10)
print(result)
"""

output = analyze_code(sample_code)
print("Code Analysis:")
for key, value in output.items():
    print(f"  {key}: {value}")