
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR PYTHON UNIT 2
PYTHON_UNIT_2_DATA = {
    "unit": "Python Unit 2: OOPs in Python",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Object-Oriented Programming (OOP).", "answer": "A programming paradigm based on the concept of 'objects', which can contain data and code."},
            {"question": "What is a Class?", "answer": "A user-defined blueprint or prototype from which objects are created."},
            {"question": "What is an Object?", "answer": "An instance of a class that has a state (attributes) and behavior (methods)."},
            {"question": "Define Encapsulation.", "answer": "The bundling of data (variables) and methods that act on the data into a single unit."},
            {"question": "What is Inheritance?", "answer": "The capability of one class to derive or inherit the properties from another class."},
            {"question": "Define Polymorphism.", "answer": "The ability of different classes to respond to the same function call in their own way."},
            {"question": "What is Abstraction?", "answer": "Hiding the complex implementation details and showing only the necessary features of the object."},
            {"question": "What is '__init__' method?", "answer": "The constructor method in Python, automatically called when a new object is created."},
            {"question": "What is 'self'?", "answer": "A reference to the current instance of the class, used to access variables that belong to the class."},
            {"question": "Define Constructor.", "answer": "A special type of method (function) which is used to initialize the instance members of the class."},
            {"question": "What are Class Variables?", "answer": "Variables that are shared by all instances of a class."},
            {"question": "What are Instance Variables?", "answer": "Variables that are unique to each instance of a class."},
            {"question": "Define Method Overriding.", "answer": "When a child class provides a specific implementation of a method that is already provided by its parent class."},
            {"question": "What is Exception Handling?", "answer": "The process of responding to unwanted or unexpected events (errors) when a computer program runs."},
            {"question": "What is 'try-except' block?", "answer": "A block of code used to catch and handle exceptions in Python."},
            {"question": "Define Assertion.", "answer": "A debugging aid that tests a condition; if the condition is false, it raises an AssertionError."},
            {"question": "What is a Binary Search?", "answer": "A search algorithm that finds the position of a target value within a sorted array."},
            {"question": "What is Sorting?", "answer": "The process of arranging data in a specific order (ascending or descending)."},
            {"question": "Define Bubble Sort.", "answer": "A simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them."},
            {"question": "What is a Hash Table?", "answer": "A data structure that implements an associative array abstract data type, a structure that can map keys to values."}
        ],
        "Part B (2-Marks)": [
            {"question": "Class vs Object.", "answer": "• Class: The blueprint/template (Logical entity). Does not consume memory.\n• Object: The instance (Physical entity). Consumes memory."},
            {"question": "Public vs Private members.", "answer": "• Public: Accessible from anywhere (name). \n• Private: Accessible only within the class (__name)."},
            {"question": "Single vs Multiple Inheritance.", "answer": "• Single: Child inherits from one Parent.\n• Multiple: Child inherits from multiple Parents (Class C(A, B))."},
            {"question": "What is Method Overloading in Python?", "answer": "• Python does not support it directly.\n• Can be simulated using default arguments or variable-length arguments."},
            {"question": "Explain 'super()' function.", "answer": "• Returns a temporary object of the superclass.\n• Used to call methods of the parent class (e.g., super().__init__())."},
            {"question": "What represents a Destructor in Python?", "answer": "• The '__del__()' method.\n• It is called when an object is garbage collected or destroyed."},
            {"question": "Linear Search vs Binary Search.", "answer": "• Linear: Check each item one by one. O(n).\n• Binary: Divide and conquer on sorted list. O(log n)."},
            {"question": "Selection Sort Logic.", "answer": "• Find the minimum element in the unsorted part.\n• Swap it with the first element of the unsorted part."},
            {"question": "What is 'raise' keyword?", "answer": "• Used to manually throw an exception.\n• Example: raise ValueError('Invalid number')."},
            {"question": "Try-Except-Finally structure.", "answer": "• Try: Code that might error.\n• Except: Code that runs if error occurs.\n• Finally: Code that ALWAYS runs (cleanup)."},
            {"question": "What is an Abstract Base Class (ABC)?", "answer": "• A class that cannot be instantiated.\n• Used to define a common interface for subclasses (module: abc)."},
            {"question": "Explain MRO (Method Resolution Order).", "answer": "• The order in which Python looks for a method in a hierarchy of classes.\n• Uses C3 Linearization algorithm (View with Class.mro())."},
            {"question": "What is Operator Overloading?", "answer": "• Giving extended meaning to operators beyond their predefined operational meaning.\n• Example: Adding two objects using '+' by implementing __add__."},
            {"question": "Bubble Sort Complexity.", "answer": "• Worst Case: O(n^2) (Reverse sorted).\n• Best Case: O(n) (Already sorted)."},
            {"question": "Usage of Assertions.", "answer": "• Used as internal self-checks for the program.\n• Syntax: assert condition, 'Error Message'."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Inheritance usage.", "answer": "1. **Definition:** Reusability mechanism.\n2. **Explanation:** Allows a class (Child) to acquire methods and properties of another class (Parent). Types: Single, Multi-level, Multiple, Hierarchical.\n3. **Example:** Class Dog(Animal): ..."},
            {"question": "Explain Encapsulation logic.", "answer": "1. **Definition:** Data hiding.\n2. **Explanation:** Restricting access to methods and variables. Prevents data from direct modification.\n3. **Example:** Using double underscore __variable to make it private."},
            {"question": "Explain Exception Handling flow.", "answer": "1. **Definition:** Error management.\n2. **Explanation:** 1. Try(Risky code). 2. Except(Handle error). 3. Else(If no error). 4. Finally(Cleanup).\n3. **Example:** Closing a file in the 'finally' block ensures it closes even if code crashes."},
            {"question": "Explain Polymorphism types.", "answer": "1. **Definition:** Many forms.\n2. **Explanation:** Compile-time (Overloading - not strict in Py) and Run-time (Overriding). formatting (+ adds nums, concats strings).\n3. **Example:** len([1,2]) vs len('abc'). Same function, different behavior."},
            {"question": "Linear Search Algorithm.", "answer": "1. **Definition:** Sequential search.\n2. **Explanation:** Iterate through the list from index 0 to n-1. Compare each element with target. If match, return index.\n3. **Example:** Finding a book on a messy shelf."},
            {"question": "Binary Search Algorithm.", "answer": "1. **Definition:** Interval search.\n2. **Explanation:** Req: Sorted list. Find middle. If target < mid, search left. If target > mid, search right. Repeat.\n3. **Example:** Looking up a word in a dictionary."},
            {"question": "Bubble Sort Algorithm.", "answer": "1. **Definition:** Sinking sort.\n2. **Explanation:** Compare adjacent pairs. If left > right, swap. Repeat pass until no swaps needed.\n3. **Example:** Bubbles (lightest elements) rise to the top."},
            {"question": "Insertion Sort Algorithm.", "answer": "1. **Definition:** Card game sort.\n2. **Explanation:** Take one element, compare with sorted sub-list, shift elements, and insert at correct position.\n3. **Example:** Sorting playing cards in your hand."},
            {"question": "User-Defined Exceptions.", "answer": "1. **Definition:** Custom errors.\n2. **Explanation:** Create a new class inheriting from correct Exception class. Raise it when specific logic fails.\n3. **Example:** class NegativeAgeError(Exception): pass."},
            {"question": "Hash Table Concepts.", "answer": "1. **Definition:** Key-Value mapping.\n2. **Explanation:** Uses a Hash Function to compute an index. Handles Collisions (Chaining/Open Addressing).\n3. **Example:** Python's 'dict' implementation."},
            {"question": "Python Magic Methods.", "answer": "1. **Definition:** Dunder methods.\n2. **Explanation:** Special methods with double underscores. __init__, __str__, __len__, __add__.\n3. **Example:** __str__ defines how object looks when printed."},
            {"question": "Benefits of OOP.", "answer": "1. **Definition:** Why use Objects?\n2. **Explanation:** Modularity for troubleshooting. Reuse of code through inheritance. Flexibility through polymorphism.\n3. **Example:** Managing large software systems efficiently."},
            {"question": "Class vs Instance Variables.", "answer": "1. **Definition:** Scope of data.\n2. **Explanation:** Class Var: Shared by all (static). Instance Var: Unique to object (self.var).\n3. **Example:** Class: Wheels=4. Instance: Color=Red."},
            {"question": "Selection Sort vs Insertion Sort.", "answer": "1. **Definition:** Comparison sorts.\n2. **Explanation:** Selection: Minimizes swaps (Find min, swap once). Insertion: Minimizes comparisons for nearly sorted data.\n3. **Example:** Insertion sort is better for live data streams."},
            {"question": "Abstract Classes.", "answer": "1. **Definition:** Interfaces.\n2. **Explanation:** Contains one or more abstract methods. Subclasses MUST implement them.\n3. **Example:** Shape class with abstract 'area()' method."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Encapsulation.",
                "answer": "1. **Definition:**\n   The fundamental concept of wrapping data (variables) and methods (functions) together as a single unit.\n\n2. **Goal:**\n   To restrict direct access to some of an object's components (Data Hiding).\n\n3. **Core Concept:**\n   Protection barrier. It prevents code outside this shield from accidentally modifying the data.\n\n4. **Technique / Method:**\n   In Python, use single underscore (_) for protected and double underscore (__) for private members.\n\n5. **Applications:**\n   Banking systems where 'Account Balance' should not be directly accessible by other modules."
            },
            {
                "question": "Detailed Explanation of Inheritance.",
                "answer": "1. **Definition:**\n   The mechanism where a new class (Check) derives the attributes and methods of an existing class (Parent).\n\n2. **Goal:**\n   Code Reusability. Write code once in Parent, use it in multiple Children.\n\n3. **Core Concept:**\n   'Is-A' relationship. A Dog IS-A Animal. A Car IS-A Vehicle.\n\n4. **Technique / Method:**\n   Syntax: `class Child(Parent):`. Supports Multi-level, Multiple, and Hierarchical inheritance.\n\n5. **Applications:**\n   Creating specific types (Manager, Developer) from a generic type (Employee)."
            },
            {
                "question": "Detailed Explanation of Polymorphism.",
                "answer": "1. **Definition:**\n   The ability of an object to take on many forms.\n\n2. **Goal:**\n   To allow a single interface to be used for a general class of actions.\n\n3. **Core Concept:**\n   Method Overriding. Using a unified method name (e.g., draw()) that behaves differently for Circle vs Square.\n\n4. **Technique / Method:**\n   Duck Typing in Python: 'If it walks like a duck and quacks like a duck, it's a duck'.\n\n5. **Applications:**\n   Graphics rendering, Plugin systems, Flexible API design."
            },
            {
                "question": "Detailed Comparison of Searching Algorithms.",
                "answer": "1. **Definition:**\n   Techniques to retrieve an element from any data structure.\n\n2. **Goal:**\n   To find the location of a target element efficiently.\n\n3. **Core Concept:**\n   Linear: Brute force, checks every item. Binary: Divide and Conquer, halves search space (requires sorted).\n\n4. **Technique / Method:**\n   Linear Time Complexity: O(n). Binary Time Complexity: O(log n).\n\n5. **Applications:**\n   Linear for small/unsorted lists. Binary for large databases/dictionaries."
            },
            {
                "question": "Detailed Explanation of Exception Handling.",
                "answer": "1. **Definition:**\n   The mechanism to handle runtime errors so the flow of the program can be maintained.\n\n2. **Goal:**\n   Robustness. To prevent the application from crashing when bad input or errors occur.\n\n3. **Core Concept:**\n   Catching errors designated as 'Exceptions'.\n\n4. **Technique / Method:**\n   `try`: code to monitor. `except`: handle specific errors. `else`: if no error. `finally`: always run.\n\n5. **Applications:**\n   File I/O (File not found), Network requests (Timeout), ZeroDivision (Math)."
            },
            {
                "question": "Detailed Explanation of Sorting Algorithms (Bubble/Selection).",
                "answer": "1. **Definition:**\n   Algorithms that put elements of a list in a certain order (Numeric/Lexicographical).\n\n2. **Goal:**\n   To organize data to make searching/retrieval faster.\n\n3. **Core Concept:**\n   Bubble: Swap adjacent. Selection: Pick min and place at start.\n\n4. **Technique / Method:**\n   Both are O(n^2) algorithms. Inefficient for large datasets but easy to implement/teach.\n\n5. **Applications:**\n   Sorting student names, organizing leaderboard scores (small scale)."
            },
            {
                "question": "Detailed Explanation of Constructors (__init__).",
                "answer": "1. **Definition:**\n   A special method of a class or structure in OOP that initializes a newly created object.\n\n2. **Goal:**\n   To assign values to the data members of the class when an object is created.\n\n3. **Core Concept:**\n   Initialization, not Creation. The object is created by `__new__` and initialized by `__init__`.\n\n4. **Technique / Method:**\n   Def `__init__(self, args):`. Self binds the attributes to the specific instance.\n\n5. **Applications:**\n   Setting up database connections or default values when an object comes to life."
            },
            {
                "question": "Detailed Explanation of Hash Tables.",
                "answer": "1. **Definition:**\n   A data structure that creates a mapping between keys and values.\n\n2. **Goal:**\n   To provide O(1) average time complexity for Lookups, Inserts, and Deletes.\n\n3. **Core Concept:**\n   Hash Function. Converts the Key into an Integer Index where the Value is stored.\n\n4. **Technique / Method:**\n   Handling Collisions: Chaining (Linked List at index) or Open Addressing (Probing).\n\n5. **Applications:**\n   Implementing Dictionaries in Python, Database Indexing, Caching."
            },
            {
                "question": "Detailed Explanation of Assertions.",
                "answer": "1. **Definition:**\n   Statements used to test if a condition in your code returns True.\n\n2. **Goal:**\n   Debugging. To instantly verify assumptions made by the program.\n\n3. **Core Concept:**\n   If condition is False, program stops with AssertionError. If True, it continues.\n\n4. **Technique / Method:**\n   `assert x > 0, 'x must be positive'`. Can be disabled globally in production (-O flag).\n\n5. **Applications:**\n   Validating internal logic, Checking invariants, Pre-conditions for functions."
            },
            {
                "question": "Detailed Explanation of Method Overriding.",
                "answer": "1. **Definition:**\n   A feature that allows a subclass to provide a specific implementation of a method already defined in its superclass.\n\n2. **Goal:**\n   Specific Behavior. To let the child class behave differently than the parent.\n\n3. **Core Concept:**\n   Runtime Polymorphism. The method called is determined by the object type at runtime.\n\n4. **Technique / Method:**\n   Define method with same name and signature in Child class. Use `super()` to extend parent logic.\n\n5. **Applications:**\n   UI Frameworks (draw() method), customized behavior for specific game entities."
            }
        ]
    }
}

def populate_python_unit2():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Python Unit 2: OOPs in Python"
        data = [u for u in data if "Python Unit 2: OOPs in Python" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(PYTHON_UNIT_2_DATA)
        print("Successfully replaced Python Unit 2 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_python_unit2()
