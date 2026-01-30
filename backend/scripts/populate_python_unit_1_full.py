
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR PYTHON UNIT 1
PYTHON_UNIT_1_DATA = {
    "unit": "Python Unit 1: Python Basics",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Python.", "answer": "Python is a high-level, interpreted commercial programming language known for readability."},
            {"question": "What is a Variable?", "answer": "A named memory location used to store data values."},
            {"question": "Define Keyword.", "answer": "Reserved words that have special meaning to the compiler/interpreter (e.g., if, else)."},
            {"question": "What is Type Casting?", "answer": "The process of converting one data type into another (e.g., int() to float())."},
            {"question": "Define String Slicing.", "answer": "Extracting a portion of a string using indices (e.g., text[0:5])."},
            {"question": "What is an Operator?", "answer": "A symbol that performs operations on variables and values."},
            {"question": "Define Indentation.", "answer": "Whitespace at the beginning of a line to define a block of code scope."},
            {"question": "What is a List?", "answer": "An ordered, mutable collection of items enclosed in square brackets []."},
            {"question": "What is a Tuple?", "answer": "An ordered, immutable collection of items enclosed in parentheses ()."},
            {"question": "Define Dictionary.", "answer": "An unordered collection of key-value pairs enclosed in curly braces {}."},
            {"question": "What is the 'input()' function?", "answer": "A built-in function used to take user input from the console as a string."},
            {"question": "What is a Comment?", "answer": "Non-executable lines used to explain code, starting with #."},
            {"question": "Define Loop.", "answer": "A control structure that repeats a block of code as long as a condition is true."},
            {"question": "What is 'break' statement?", "answer": "Used to exit the nearest enclosing loop immediately."},
            {"question": "What is 'continue' statement?", "answer": "Used to skip the rest of the code inside a loop for the current iteration."},
            {"question": "Define Function.", "answer": "A block of organized, reusable code that performs a single, related action."},
            {"question": "What is a Module?", "answer": "A file containing Python definitions and statements (e.g., .py file)."},
            {"question": "What is 'import'?", "answer": "Keyword used to bring code from external modules into the current script."},
            {"question": "Define Syntax Error.", "answer": "An error that occurs when the parser detects an incorrect statement in the code."},
            {"question": "What is IDLE?", "answer": "Integrated Development and Learning Environment; a basic IDE for Python."}
        ],
        "Part B (2-Marks)": [
            {"question": "List 4 Data Types in Python.", "answer": "• Integer (int)\n• Float (float)\n• String (str)\n• Boolean (bool)"},
            {"question": "List vs Tuple.", "answer": "• List: Mutable, uses [ ], slower.\n• Tuple: Immutable, uses ( ), faster."},
            {"question": "Difference between / and //.", "answer": "• / (Float Division): Returns float result (e.g., 5/2 = 2.5).\n• // (Floor Division): Returns integer floor (e.g., 5//2 = 2)."},
            {"question": "Explain 'is' vs '==' operator.", "answer": "• '==': Checks if values are equal.\n• 'is': Checks if operands refer to the exact same object in memory."},
            {"question": "What are Mutable types?", "answer": "• Objects whose state can be modified after creation.\n• Examples: List, Dictionary, Set."},
            {"question": "What are Immutable types?", "answer": "• Objects whose state cannot be modified after creation.\n• Examples: Int, Float, String, Tuple."},
            {"question": "Explain Logical Operators.", "answer": "• 'and': True if both are true.\n• 'or': True if at least one is true.\n• 'not': Inverts the boolean value."},
            {"question": "What is a Docstring?", "answer": "• A string literal used to document a function or class.\n• Enclosed in triple quotes \"\"\"...\"\"\". "},
            {"question": "Local vs Global Variables.", "answer": "• Local: Declared inside a function, accessible only there.\n• Global: Declared outside, accessible throughout the module."},
            {"question": "What is a Lambda Function?", "answer": "• An anonymous, small function defined with 'lambda'.\n• Syntax: lambda arguments : expression."},
            {"question": "Dictionary keys() vs values().", "answer": "• keys(): Returns a view object of all keys.\n• values(): Returns a view object of all values."},
            {"question": "What is 'pass' statement?", "answer": "• A null operation; nothing happens when it executes.\n• Used as a placeholder for future code."},
            {"question": "Explain Range() function.", "answer": "• Generates a sequence of numbers.\n• Syntax: range(start, stop, step)."},
            {"question": "Open() modes 'r' vs 'w'.", "answer": "• 'r': Read mode (Default). Error if file doesn't exist.\n• 'w': Write mode. Creates file if not exists, overwrites if it does."},
            {"question": "What is String Formatting?", "answer": "• Inserting variables into strings.\n• Methods: f-strings (f'Age: {age}'), .format(), and % operator."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Arithmetic Operators.", "answer": "1. **Definition:** Mathematical operations.\n2. **Explanation:** + (Add), - (Sub), * (Mul), / (Div), % (Modulus), ** (Exponent), // (Floor Div).\n3. **Example:** 10 % 3 = 1 (Remainder)."},
            {"question": "Explain Membership Operators.", "answer": "1. **Definition:** Testing for membership in a sequence.\n2. **Explanation:** 'in' (True if found), 'not in' (True if not found).\n3. **Example:** 'a' in 'apple' is True."},
            {"question": "Conditional Statements Structure.", "answer": "1. **Definition:** Decision making.\n2. **Explanation:** if condition: code; elif condition: code; else: code.\n3. **Example:** if x > 0: print('Positive')."},
            {"question": "Explain While Loop.", "answer": "1. **Definition:** Repeats code while condition is True.\n2. **Explanation:** Checks condition -> Executes block -> Repeats. Infinite loop if condition never False.\n3. **Example:** while x < 5: x += 1."},
            {"question": "Explain For Loop.", "answer": "1. **Definition:** Iterates over a sequence.\n2. **Explanation:** Used for iterating lists, tuples, strings, or range().\n3. **Example:** for i in range(5): print(i)."},
            {"question": "String Methods Checklist.", "answer": "1. **Definition:** Built-in string manipulations.\n2. **Explanation:** .upper(), .lower(), .strip() (removes whitespace), .replace(old, new).\n3. **Example:** ' hello '.strip() -> 'hello'."},
            {"question": "List Methods Checklist.", "answer": "1. **Definition:** Built-in list manipulations.\n2. **Explanation:** .append() (add to end), .insert() (add at index), .pop() (remove), .sort().\n3. **Example:** mylist.append(10)."},
            {"question": "Dictionary Methods Checklist.", "answer": "1. **Definition:** Built-in dict manipulations.\n2. **Explanation:** .get(key), .keys(), .values(), .items(), .update().\n3. **Example:** d.get('name', 'Unknown')."},
            {"question": "Defining a Function.", "answer": "1. **Definition:** Creating reusable logic.\n2. **Explanation:** Use 'def' keyword, followed by name, parentheses (params), and colon.\n3. **Example:** def greet(name): return f'Hi {name}'."},
            {"question": "File Handling Steps.", "answer": "1. **Definition:** Reading/Writing files.\n2. **Explanation:** 1. Open file (open()). 2. Process (read/write). 3. Close (close()).\n3. **Example:** with open('f.txt') as f: data = f.read()."},
            {"question": "Arguments: Positional vs Keyword.", "answer": "1. **Definition:** Ways to pass arguments.\n2. **Explanation:** Positional: Order matters (func(1, 2)). Keyword: Name matters (func(a=1, b=2)).\n3. **Example:** print(sep='-', end='')."},
            {"question": "Exception Handling Basics.", "answer": "1. **Definition:** Managing runtime errors.\n2. **Explanation:** Use try: block for risky code, except: block to catch error.\n3. **Example:** try: x = 1/0 except ZeroDivisionError: print('Error')."},
            {"question": "Return vs Print.", "answer": "1. **Definition:** Output mechanisms.\n2. **Explanation:** Print displays to console (Side effect). Return sends value back to caller (Data flow).\n3. **Example:** x = print(5) -> x is None. x = return 5 -> x is 5."},
            {"question": "Negative Indexing.", "answer": "1. **Definition:** Accessing from the end.\n2. **Explanation:** -1 is last item, -2 is second last.\n3. **Example:** list[-1] gets the last element."},
            {"question": "Slicing Syntax.", "answer": "1. **Definition:** Extracting subsequences.\n2. **Explanation:** sequence[start:stop:step]. Stop is exclusive.\n3. **Example:** 'Python'[0:2] -> 'Py'."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Python Features.",
                "answer": "1. **Definition:**\n   Characteristics that make Python a popular programming language.\n\n2. **Goal:**\n   To provide a language that is powerful yet easy to read and write.\n\n3. **Core Concept:**\n   Key features: Interpreted, Dynamically Typed, Object-Oriented, High-Level, Excessive Library Support.\n\n4. **Technique / Method:**\n   No compilation step needed. Variables don't need type declaration. Supports Classes/Polymorphism.\n\n5. **Applications:**\n   Web Dev (Django), Data Science (Pandas), AI (TensorFlow), Automation."
            },
            {
                "question": "Detailed Explanation of Operators.",
                "answer": "1. **Definition:**\n   Special symbols in Python that carry out arithmetic or logical computation.\n\n2. **Goal:**\n   To manipulate individual data items and return a result.\n\n3. **Core Concept:**\n   Types: Arithmetic (+,-), Relational (>,<), Logical (and,or), Assignment (=), Bitwise, Membership.\n\n4. **Technique / Method:**\n   Operator Precedence (PEMDAS) determines the order of execution in complex expressions.\n\n5. **Applications:**\n   Calculating totals, boolean logic checks in loops, flag settings."
            },
            {
                "question": "Detailed Explanation of Control Flow (If-Else).",
                "answer": "1. **Definition:**\n   The order in which individual statements, instructions, or function calls are executed.\n\n2. **Goal:**\n   To allow the program to make decisions and execute different code paths.\n\n3. **Core Concept:**\n   Conditional branching. If condition is True -> Do A. Else -> Do B.\n\n4. **Technique / Method:**\n   Keywords: if, elif, else. Requires indentation to define blocks. Nested ifs supported.\n\n5. **Applications:**\n   Login validation (Password correct?), Game over checks, Grading systems."
            },
            {
                "question": "Detailed Explanation of Loops (For & While).",
                "answer": "1. **Definition:**\n   Structures that repeat a sequence of instructions until a specific condition is met.\n\n2. **Goal:**\n   To automate repetitive tasks without writing redundant code.\n\n3. **Core Concept:**\n   For loop: Iterates over a sequence (fixed times). While loop: Runs until condition False (variable times).\n\n4. **Technique / Method:**\n   Loop Control: break (exit), continue (skip), pass (empty). usage of iterables.\n\n5. **Applications:**\n   Processing items in a shopping cart, Reading lines in a file, server listening loops."
            },
            {
                "question": "Detailed Explanation of Functions.",
                "answer": "1. **Definition:**\n   A block of code which only runs when it is called.\n\n2. **Goal:**\n   To maximize code reusability and modularity.\n\n3. **Core Concept:**\n   Parameters (Inputs) -> Function Body (Process) -> Return Value (Output).\n\n4. **Technique / Method:**\n   Defined using `def`. Can have Default args, Keyword args, and Variable-length args (*args).\n\n5. **Applications:**\n   Mathematical formulas, API handlers, Utility helpers (e.g., date formatting)."
            },
            {
                "question": "Detailed Explanation of Lists.",
                "answer": "1. **Definition:**\n   One of 4 built-in data types in Python used to store collections of data.\n\n2. **Goal:**\n   To store multiple items in a single variable dynamically.\n\n3. **Core Concept:**\n   Ordered, Mutable, Allow Duplicates. Indexed starting at 0.\n\n4. **Technique / Method:**\n   Syntax: `mylist = [1, 2, 'a']`. Methods: append, remove, pop, sort, reverse.\n\n5. **Applications:**\n   Storing database records, queue management, stacking items."
            },
            {
                "question": "Detailed Explanation of Dictionaries.",
                "answer": "1. **Definition:**\n   A collection which is unordered, changeable, and indexed.\n\n2. **Goal:**\n   To map keys to values for fast lookup.\n\n3. **Core Concept:**\n   Key-Value pairs. Keys must be unique and immutable. Values can be anything.\n\n4. **Technique / Method:**\n   Syntax: `d = {'name': 'John', 'age': 30}`. Access: `d['name']`. Methods: keys(), values().\n\n5. **Applications:**\n   JSON data handling, User profiles, Configuration settings."
            },
            {
                "question": "Detailed Explanation of String Manipulation.",
                "answer": "1. **Definition:**\n   Operations performed on string objects to modify, parse, or format them.\n\n2. **Goal:**\n   To process textual data efficiently.\n\n3. **Core Concept:**\n   Strings are immutable arrays of bytes representing unicode characters.\n\n4. **Technique / Method:**\n   Slicing [start:end], Concatenation (+), formatting (f-strings), methods (split, join, strip).\n\n5. **Applications:**\n   Data cleaning, URL parsing, Chatbot text processing."
            },
            {
                "question": "Detailed Explanation of File Handling.",
                "answer": "1. **Definition:**\n   The ability of a program to read from and write to permanent storage files.\n\n2. **Goal:**\n   To persist data beyond the runtime of the program.\n\n3. **Core Concept:**\n   Modes: 'r' (read), 'w' (write), 'a' (append). Context Managers (`with`) ensure safety.\n\n4. **Technique / Method:**\n   `with open('file.txt', 'w') as f: f.write('Hi')`. Automatically closes file.\n\n5. **Applications:**\n   Logging errors, Saving user settings, Exporting reports to CSV."
            },
            {
                "question": "Detailed Explanation of Modules and Packages.",
                "answer": "1. **Definition:**\n   The modular programming approach in Python.\n\n2. **Goal:**\n   To organize code into logical, manageable, and separate files/folders.\n\n3. **Core Concept:**\n   Module: Single .py file. Package: Folder with __init__.py containing modules.\n\n4. **Technique / Method:**\n   Using `import math` or `from math import sqrt`. PyPI (pip) manages external packages.\n\n5. **Applications:**\n   Using standard libraries (os, sys, datetime) or third-party (requests, numpy)."
            }
        ]
    }
}

def populate_python_unit1():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Python Unit 1: Python Basics"
        data = [u for u in data if "Python Unit 1: Python Basics" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(PYTHON_UNIT_1_DATA)
        print("Successfully replaced Python Unit 1 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_python_unit1()
