
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR PYTHON UNIT 3
PYTHON_UNIT_3_DATA = {
    "unit": "Python Unit 3: Plotting & Algorithms",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "What is PyLab?", "answer": "A module that bundles numpy and matplotlib in a single namespace to provide a Matlab-like environment."},
            {"question": "Define Matplotlib.", "answer": "A plotting library for the Python programming language and its numerical mathematics extension NumPy."},
            {"question": "What is a Line Plot?", "answer": "A type of chart which displays information as a series of data points connected by straight line segments."},
            {"question": "What is a Bar Chart?", "answer": "A chart that presents categorical data with rectangular bars with heights proportional to the values."},
            {"question": "What is the Fibonacci Sequence?", "answer": "A series of numbers where each number is the sum of the two preceding ones (0, 1, 1, 2, 3...)."},
            {"question": "Define Recursion.", "answer": "A method of solving a problem where the solution depends on solutions to smaller instances of the same problem."},
            {"question": "What is Dynamic Programming?", "answer": "An optimization method that solves complex problems by breaking them into overlapping subproblems and storing results."},
            {"question": "Define Memoization.", "answer": "An optimization technique used to cache the result of expensive function calls and return the cached result when the same inputs occur again."},
            {"question": "What is the 0/1 Knapsack Problem?", "answer": "A problem where items with weights and values must be selected to maximize value without exceeding capacity."},
            {"question": "Define Divide and Conquer.", "answer": "An algorithm design paradigm that recursively breaks a problem into two or more sub-problems of the same type."},
            {"question": "What is 'pyplot'?", "answer": "A collection of command style functions in matplotlib (import matplotlib.pyplot as plt)."},
            {"question": "Define Optimization Problem.", "answer": "A problem of finding the best solution from all feasible solutions."},
            {"question": "What is Greedy Algorithm?", "answer": "An algorithm that makes the locally optimal choice at each stage with the hope of finding a global optimum."},
            {"question": "What is base case in recursion?", "answer": "The condition that stops the recursion from continuing infinitely."},
            {"question": "What is Big O Notation?", "answer": "A mathematical notation that describes the limiting behavior of a function (Algorithm Complexity)."},
            {"question": "What is Histogram?", "answer": "A graphical representation of the distribution of numerical data using bars."},
            {"question": "Define Scatter Plot.", "answer": "A graph that uses dots to represent values for two different numeric variables."},
            {"question": "What is 'legend' in plotting?", "answer": "An area of the graph describing the elements of the graph (what each line/color represents)."},
            {"question": "Difference between list and numpy array.", "answer": "Numpy arrays are more compact, faster, and support vectorization compared to Python lists."},
            {"question": "What is Tree Traversal?", "answer": "The process of visiting (checking/updating) each node in a tree data structure exactly once."}
        ],
        "Part B (2-Marks)": [
            {"question": "Memoization vs Tabulation.", "answer": "• Memoization: Top-Down approach (Recursion + Caching).\n• Tabulation: Bottom-Up approach (Iterative table filling)."},
            {"question": "Recursive vs Iterative.", "answer": "• Recursive: Calls itself, uses Stack memory, cleaner code.\n• Iterative: Uses loops, efficient memory usage, harder to implement complex logic."},
            {"question": "Line Plot vs Scatter Plot.", "answer": "• Line: Connects points (shows trends over time).\n• Scatter: Individual points (shows correlation between variables)."},
            {"question": "Explain 'xlabel' and 'ylabel'.", "answer": "• xlabel(): Sets the label for the X-axis.\n• ylabel(): Sets the label for the Y-axis."},
            {"question": "Divide & Conquer vs Dynamic Programming.", "answer": "• D&C: Subproblems are independent (e.g., Merge Sort).\n• DP: Subproblems overlap (e.g., Fibonacci)."},
            {"question": "What is 'grid()' in plots?", "answer": "• A function to turn the grid lines on or off.\n• Helps in reading values from the chart easier."},
            {"question": "Complexity of Fibonacci (Recursive).", "answer": "• Exponential Time: O(2^n).\n• Very inefficient without memoization."},
            {"question": "Knapsack: Fractional vs 0/1.", "answer": "• 0/1: Cannot break items (Take or Leave). Solved by DP.\n• Fractional: Can take part of item. Solved by Greedy."},
            {"question": "What is 'subplot'?", "answer": "• A function to create multiple plots in a single figure window.\n• Syntax: subplot(nrows, ncols, index)."},
            {"question": "Advantages of NumPy.", "answer": "• Fast vectorized operations.\n• Efficient memory use.\n• Broadcasting capabilities."},
            {"question": "Optimal Substructure Property.", "answer": "• A problem has this if an optimal solution can be constructed from optimal solutions of its subproblems."},
            {"question": "Overlapping Subproblems Property.", "answer": "• A problem has this if recursive algorithm visits the same subproblems repeatedly."},
            {"question": "What is 'title()' in plotting?", "answer": "• Sets the title of the current plot.\n• plt.title('Sales Data')."},
            {"question": "Merge Sort Complexity.", "answer": "• Time Complexity: O(n log n) in all cases.\n• Space Complexity: O(n) (Requires aux array)."},
            {"question": "What is 'show()' function?", "answer": "• Display the figure window.\n• Must be called at the end of the script to see the plot."}
        ],
        "Part C (3-Marks)": [
            {"question": "Steps to Plot a Graph.", "answer": "1. **Definition:** Visualization workflow.\n2. **Explanation:** 1. Import pyplot. 2. Define Data (x, y lists). 3. Plot (plt.plot(x,y)). 4. Add Labels/Title. 5. Show (plt.show()).\n3. **Example:** Plotting y = x^2."},
            {"question": "Fibonacci with Memoization.", "answer": "1. **Definition:** Optimizing recursion.\n2. **Explanation:** Store computed Fib(n) in a dict/array. Before computing, check if n is in cache. If yes, return it.\n3. **Example:** Turns O(2^n) into O(n)."},
            {"question": "0/1 Knapsack Logic.", "answer": "1. **Definition:** Constraint optimization.\n2. **Explanation:** For each item, decide: Include it (add value, reduce capacity) OR Exclude it (keep capacity). Maximize value.\n3. **Example:** Thief choosing jewels to fit in a bag."},
            {"question": "Divide and Conquer Strategy.", "answer": "1. **Definition:** Algorithm paradigm.\n2. **Explanation:** 1. Divide (Break problem). 2. Conquer (Solve subproblems recursively). 3. Combine (Merge solutions).\n3. **Example:** Merge Sort or Quick Sort."},
            {"question": "Bar Chart Usage.", "answer": "1. **Definition:** Categorical comparison.\n2. **Explanation:** Uses geometric bars to compare distinct groups. Height = Value.\n3. **Example:** Comparing population of 5 different cities."},
            {"question": "Features of PyLab.", "answer": "1. **Definition:** Scientific computing env.\n2. **Explanation:** Combines NumPy (Math) + Scipy (Science) + Matplotlib (Plotting). Designed for interactive prototyping.\n3. **Example:** Rapidly testing signal processing algorithms."},
            {"question": "Depth First vs Breadth First Search.", "answer": "1. **Definition:** Graph traversal.\n2. **Explanation:** DFS: Explores as deep as possible (Stack). BFS: Explores neighbor nodes first (Queue).\n3. **Example:** DFS for maze solving. BFS for shortest path."},
            {"question": "Binary Search Recursive.", "answer": "1. **Definition:** Recursive search.\n2. **Explanation:** Base case: not found or found. Recursive step: call binary_search on left or right half.\n3. **Example:** Elegant way to write binary search."},
            {"question": "Advantages of Dynamic Programming.", "answer": "1. **Definition:** Efficiency.\n2. **Explanation:** drastic reduction in time complexity for problems with overlapping subproblems. Trade space (memory) for time.\n3. **Example:** Calculating Fibonacci(100) instantly."},
            {"question": "Saving Plots.", "answer": "1. **Definition:** Exporting visualization.\n2. **Explanation:** Use `plt.savefig('filename.png')`. Supports png, pdf, svg. Resolution controlled by dpi param.\n3. **Example:** Saving high-res charts for a paper."},
            {"question": "Customizing Plots.", "answer": "1. **Definition:** Styling.\n2. **Explanation:** Change color ('r'), marker ('o'), linestyle ('--'), linewidth. Add grid, legend, annotations.\n3. **Example:** plt.plot(x, y, 'r--', linewidth=2)."},
            {"question": "Greedy vs DP.", "answer": "1. **Definition:** Approach comparison.\n2. **Explanation:** Greedy: Make best choice NOW (fails in global optimization sometimes). DP: Consider ALL choices and store results.\n3. **Example:** Greedy fails Coin Change for [1,3,4] and target 6 (3+3 is best, Greedy picks 4+1+1)."},
            {"question": "Merge Sort Logic.", "answer": "1. **Definition:** Stable sort.\n2. **Explanation:** Recursively split list in half until size 1. Merge sorted halves back together in order.\n3. **Example:** Sorting a deck of cards by splitting piles."},
            {"question": "Time Complexity Analysis.", "answer": "1. **Definition:** Efficiency measurement.\n2. **Explanation:** O(1) Constant, O(log n) Logarithmic, O(n) Linear, O(n^2) Quadratic. Count basic operations.\n3. **Example:** Nested loops usually imply O(n^2)."},
            {"question": "Base Case Importance.", "answer": "1. **Definition:** Termination condition.\n2. **Explanation:** Without it, recursion causes Stack Overflow Error. Must be reached eventually.\n3. **Example:** Factorial(0) = 1."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Dynamic Programming.",
                "answer": "1. **Definition:**\n   An algorithmic technique for solving an optimization problem by breaking it down into simpler subproblems.\n\n2. **Goal:**\n   To solve problems with Overlapping Subproblems and Optimal Substructure efficiently.\n\n3. **Core Concept:**\n   'Remembering the past'. Store results of subproblems (Memoization/Tabulation) to avoid re-computing.\n\n4. **Technique / Method:**\n   1. Characterize structure of optimal solution. 2. Define value recursively. 3. Compute value (Bottom-up usually).\n\n5. **Applications:**\n   Shortest Path (Bellman-Ford), Text Diff (LCS), DNA Sequence alignment, Resource Allocation."
            },
            {
                "question": "Detailed Explanation of 0/1 Knapsack Problem.",
                "answer": "1. **Definition:**\n   A combinatorial optimization problem. Given a set of items with weights/values, determine items to include in a collection.\n\n2. **Goal:**\n   To maximize the total value while keeping total weight less than or equal to a given limit.\n\n3. **Core Concept:**\n   Constraint Satisfaction. You cannot break items (0 or 1). Greedy fails here.\n\n4. **Technique / Method:**\n   Use DP. Create a 2D table `K[i][w]`. `max(val + K[i-1][w-wt], K[i-1][w])`.\n\n5. **Applications:**\n   Resource management, Portfolio selection in finance, Cargo loading."
            },
            {
                "question": "Detailed Explanation of Divide and Conquer.",
                "answer": "1. **Definition:**\n   An algorithm design paradigm based on multi-branched recursion.\n\n2. **Goal:**\n   To solve a problem by dividing it into smaller sub-problems until they are simple enough to be solved directly.\n\n3. **Core Concept:**\n   Divide (Break) -> Conquer (Solve) -> Combine (Merge).\n\n4. **Technique / Method:**\n   Recursion is essential. Problems must be independent.\n\n5. **Applications:**\n   Merge Sort, Quick Sort, Binary Search, Matrix Multiplication (Strassen's)."
            },
            {
                "question": "Detailed Explanation of Recursion.",
                "answer": "1. **Definition:**\n   A programming technique where a function calls itself directly or indirectly.\n\n2. **Goal:**\n   To solve problems that have a self-similar structure (e.g., Tree traversal, Inductive definitions).\n\n3. **Core Concept:**\n   Base Case (Stop condition) + Recursive Case (Reduction step).\n\n4. **Technique / Method:**\n   `def f(n): if n==0: return 1 else: return n * f(n-1)`.\n\n5. **Applications:**\n   DFS, Factorial, Towers of Hanoi, Fractal generation."
            },
            {
                "question": "Detailed Explanation of Matplotlib/PyLab.",
                "answer": "1. **Definition:**\n   A comprehensive library for creating static, animated, and interactive visualizations in Python.\n\n2. **Goal:**\n   To make complex data understandable through visual representation (Charts/Graphs).\n\n3. **Core Concept:**\n   Figure (Canvas) and Axes (Plot area). PyLab interface mimics MATLAB for ease of use.\n\n4. **Technique / Method:**\n   `import matplotlib.pyplot as plt`. `plt.plot()`, `plt.bar()`, `plt.scatter()`. Customization via methods.\n\n5. **Applications:**\n   Data Science EDA (Exploratory Data Analysis), Financial Charting, Scientific reporting."
            },
            {
                "question": "Detailed Explanation of Fibonacci Sequence (DP vs Recursive).",
                "answer": "1. **Definition:**\n   A series of numbers where the next number is found by adding up the two numbers before it.\n\n2. **Goal:**\n   To calculate the Nth number efficiently.\n\n3. **Core Concept:**\n   Recursive: O(2^n) (Bad, repeats work). DP: O(n) (Good, reuses work).\n\n4. **Technique / Method:**\n   Iterative: `a, b = 0, 1; for _ in range(n): a, b = b, a+b`.\n\n5. **Applications:**\n   Modeling population growth, financial markets, computing algorithms benchmark."
            },
            {
                "question": "Detailed Explanation of Greedy Algorithm.",
                "answer": "1. **Definition:**\n   An algorithmic paradigm that builds up a solution piece by piece.\n\n2. **Goal:**\n   To find the global optimum by choosing the local optimum at each step.\n\n3. **Core Concept:**\n   Short-sightedness. Make the best decision for NOW. Do not look ahead.\n\n4. **Technique / Method:**\n   Sort items -> Pick best available -> Repeat. Simple and fast.\n\n5. **Applications:**\n   Dijkstra's Algorithm, Huffman Coding, Activity Selection Problem."
            },
            {
                "question": "Detailed Explanation of Plot Types (Line, Bar, Scatter).",
                "answer": "1. **Definition:**\n   Different ways to represent data visually depending on the data nature.\n\n2. **Goal:**\n   To choose the right chart for the right message.\n\n3. **Core Concept:**\n   Line: Continuous data (Time series). Bar: Discrete categories (Comparison). Scatter: Correlation (XY data).\n\n4. **Technique / Method:**\n   `plt.plot()` for Line. `plt.bar()` for Bar. `plt.scatter()` for Scatter.\n\n5. **Applications:**\n   Stock prices (Line), Population (Bar), Height vs Weight (Scatter)."
            },
            {
                "question": "Detailed Explanation of Merge Sort.",
                "answer": "1. **Definition:**\n   An efficient, stability-preserving, comparison-based sorting algorithm.\n\n2. **Goal:**\n   To sort a list of N elements in O(N log N) time reliably.\n\n3. **Core Concept:**\n   Divide and Conquer. Split list into halves until atomic. Merge sorted halves.\n\n4. **Technique / Method:**\n   Recursive function `mergeSort(arr)`. Helper function `merge(left, right)`.\n\n5. **Applications:**\n   Sorting Linked Lists, E-commerce product sorting, External Sorting (Large files)."
            },
            {
                "question": "Detailed Explanation of Complexity (Big O).",
                "answer": "1. **Definition:**\n   A mathematical concept used to describe the performance or complexity of an algorithm.\n\n2. **Goal:**\n   To analyze how runtime or space requirements grow as input size (N) grows.\n\n3. **Core Concept:**\n   Worst-case scenario analysis. Ignore constants, focus on growth rate.\n\n4. **Technique / Method:**\n   O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n).\n\n5. **Applications:**\n   Selecting the right algorithm for large datasets (e.g., Choosing QuickSort over BubbleSort)."
            }
        ]
    }
}

def populate_python_unit3():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Python Unit 3: Plotting & Algorithms"
        data = [u for u in data if "Python Unit 3: Plotting & Algorithms" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(PYTHON_UNIT_3_DATA)
        print("Successfully replaced Python Unit 3 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_python_unit3()
