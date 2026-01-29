import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# EXPERT CONTENT FOR ALL 10 UNITS
ALL_EXPERT_DATA = [
     # ---------------- PYTHON UNIT 1 ----------------
    {
        "unit": "Unit 1: Python Basics",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define Python.", "answer": "Python is a high-level, interpreted programmimg language."},
                {"question": "What is IDLE?", "answer": "Integrated Development and Learning Environment for Python."},
                {"question": "Define Variable.", "answer": "A reserved memory location to store values."},
                {"question": "What is a Keyword?", "answer": "Reserved words that have special meaning to the compiler."},
                {"question": "What is a Comment?", "answer": "Lines ignored by the interpreter, used for documentation."},
                {"question": "Define Indentation.", "answer": "Whitespace used to define the scope of loops and functions."},
                {"question": "What is Type Casting?", "answer": "Converting a variable from one data type to another."},
                {"question": "What is a List?", "answer": "Ordered, mutable collection of items."},
                {"question": "What is a Tuple?", "answer": "Ordered, immutable collection of items."},
                {"question": "What is a Dictionary?", "answer": "Unordered collection of Key-Value pairs."},
                {"question": "What is a Set?", "answer": "Unordered collection of unique items."},
                {"question": "Define Operator.", "answer": "Symbol that performs operations on variables/values."},
                {"question": "What is 'break'?", "answer": "Statement used to exit a loop immediately."},
                {"question": "What is 'continue'?", "answer": "Statement used to skip current iteration and move to next."},
                {"question": "What is a Function?", "answer": "Block of organized code that performs a specific task."}
            ],
            "Part B (2-Marks)": [
                {"question": "List 4 Features of Python.", "answer": "• Easy to Learn\n• Interpreted Language\n• Object-Oriented\n• Huge Library Support"},
                {"question": "Difference between List and Tuple.", "answer": "• List: Mutable (Can change), Uses [ ].\n• Tuple: Immutable (Cannot change), Uses ( )."},
                {"question": "Explain 'is' and 'in' operators.", "answer": "• 'is' (Identity): Checks if objects are same memory location.\n• 'in' (Membership): Checks if value exists in sequence."},
                {"question": "What are Mutable and Immutable types?", "answer": "• Mutable: Can be changed (List, Dict, Set).\n• Immutable: Cannot be changed (Int, Float, String, Tuple)."},
                {"question": "Difference between / and // operators.", "answer": "• / (Float Division): Returns float (5/2 = 2.5).\n• // (Floor Division): Returns integer floor (5//2 = 2)."},
                {"question": "What is Docstring?", "answer": "• String literal used to document a function/class.\n• Enclosed in triple quotes \"\"\"Doc\"\"\". "},
                {"question": "Explain 'pass' statement.", "answer": "• Null statement. Nothing happens when executed.\n• Used as placeholder for future code loops/functions."},
                {"question": "Global vs Local Variables.", "answer": "• Local: Defined inside function, accessible only there.\n• Global: Defined outside, accessible everywhere."},
                {"question": "Python 2 vs Python 3.", "answer": "• Print: Statement in Py2, Function in Py3.\n• Division: Integer in Py2 (5/2=2), Float in Py3 (5/2=2.5)."},
                {"question": "What is Lambda function?", "answer": "• Anonymous function defined without a name.\n• Syntax: lambda arguments: expression"}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Python Data Types with examples.", "answer": "1. **Numeric:** int (10), float (10.5), complex (1+2j).\n2. **Sequence:** list ([1,2]), tuple ((1,2)), range.\n3. **Mapping:** dict ({'a':1}).\n4. **Set:** set ({1,2})."},
                {"question": "Explain Control Flow Statements.", "answer": "1. **Conditional:** if, elif, else (Decision making).\n2. **Looping:** for, while (Iteration).\n3. **Transfer:** break, continue, pass."},
                {"question": "Discuss Operator Precedence.", "answer": "1. **Definition:** Order in which operations are performed.\n2. **Order:** parenthesis () -> exponent ** -> mul/div * / -> add/sub + -.\n3. **Example:** 2 + 3 * 5 = 17 (not 25)."},
                {"question": "Explain Dictionary Methods.", "answer": "1. **keys():** Returns all keys.\n2. **values():** Returns all values.\n3. **items():** Returns (key, value) pairs.\n4. **get(key):** Returns value safely."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain Operators in Python in detail.", "answer": "1. **Arithmetic:** +, -, *, /, %, **.\n2. **Comparison:** ==, !=, >, <, >=.\n3. **Logical:** and, or, not.\n4. **Assignment:** =, +=, -=.\n5. **Membership:** in, not in.\n6. **Identity:** is, is not.\n7. **Bitwise:** &, |, ^."},
                {"question": "Explain Functions in Python.", "answer": "1. **Definition:** Reusable block of code.\n2. **Syntax:** def name(params):\n3. **Arguments:** Positional, Keyword, Default, Variable-length (*args).\n4. **Return:** Sending value back.\n5. **Lambda:** Small anonymous functions."}
            ]
        }
    },
    
    # ---------------- PYTHON UNIT 2 ----------------
    {
        "unit": "Unit 2: OOPs in Python",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define Class.", "answer": "A blueprint for creating objects."},
                {"question": "Define Object.", "answer": "An instance of a class."},
                {"question": "What is Inheritance?", "answer": "Mechanism where a new class inherits properties of an existing class."},
                {"question": "What is Polymorphism?", "answer": "Ability to take multiple forms (Same method name, different behavior)."},
                {"question": "What is Encapsulation?", "answer": "Wrapping data and methods into a single unit."},
                {"question": "What is Abstraction?", "answer": "Hiding implementation details and showing only functionality."},
                {"question": "What is 'self'?", "answer": "Reference to the current instance of the class."},
                {"question": "What is __init__?", "answer": "Constructor method used to initialize objects."},
                {"question": "What is Method Overriding?", "answer": "Child class providing specific implementation of a parent method."},
                {"question": "What is Method Overloading?", "answer": "Same method name with different parameters (Not directly supported in Python)."},
                {"question": "What is a Destructor?", "answer": "Method (__del__) called when object is destroyed."},
                {"question": "Public vs Private.", "answer": "Public: Accessible everywhere. Private: Accessible only inside class."},
                {"question": "What is a Module?", "answer": "A file containing Python code (functions, classes)."},
                {"question": "What is a Package?", "answer": "A directory containing Python modules and __init__.py."},
                {"question": "What is Multiple Inheritance?", "answer": "A child class inheriting from more than one parent class."}
            ],
            "Part B (2-Marks)": [
                {"question": "Class vs Object.", "answer": "• Class: Logical template (e.g., 'Car Blueprint').\n• Object: Physical entity (e.g., 'Red Ferrari')."},
                {"question": "Explain __init__ method.", "answer": "• Constructor method called automatically upon object creation.\n• Used to set initial values for object attributes."},
                {"question": "Types of Inheritance.", "answer": "• Single\n• Multiple\n• Multilevel\n• Hierarchical\n• Hybrid"},
                {"question": "What are Access Specifiers?", "answer": "• Public: name (Accessible all)\n• Protected: _name (Subclasses)\n• Private: __name (Class only)"},
                {"question": "Explain 'super()' function.", "answer": "• Used to call methods of the parent class.\n• Useful in inheritance to access parent logic."},
                {"question": "Method Overloading vs Overriding.", "answer": "• Overloading: Compile-time (Same name, diff args - Not in Python).\n• Overriding: Run-time (Same name, diff implementation in Child)."},
                {"question": "What is Data Hiding?", "answer": "• Prevents direct access to data.\n• Achieved using private variables (__var) in Python."},
                {"question": "Class Variable vs Instance Variable.", "answer": "• Class Var: Shared by all instances (Static).\n• Instance Var: Unique to each object (self.var)."},
                {"question": "What is an Abstract Class?", "answer": "• Class that cannot be instantiated.\n• Contains abstract methods that strict subclasses must implement."},
                {"question": "What is Exception Handling?", "answer": "• Managing errors gracefully using try, except blocks.\n• Prevents program crash."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Inheritance with example.", "answer": "1. **Definition:** Parent -> Child relationship.\n2. **Syntax:** class Child(Parent):\n3. **Benefit:** Code reusability."},
                {"question": "Explain Polymorphism with example.", "answer": "1. **Definition:** One interface, many forms.\n2. **Example:** Duck Typing. A function can take any object that has a 'speak()' method.\n3. **Operators:** '+' adds numbers but concatenates strings (Overloading)."},
                {"question": "Explain Exception Handling blocks.", "answer": "1. **try:** Critical code.\n2. **except:** Error handling code.\n3. **finally:** Code that runs always (Cleanup).\n4. **raise:** Triggering an error manually."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain the 4 Pillars of OOPs.", "answer": "1. **Encapsulation:** Binding data+code. data hiding.\n2. **Abstraction:** Hiding complexity, showing interface.\n3. **Inheritance:** Code reuse, Hierarchy.\n4. **Polymorphism:** Flexibility, Overriding, Overloading."},
                {"question": "Explain Types of Inheritance in detail.", "answer": "1. **Single:** A->B\n2. **Multilevel:** A->B->C\n3. **Multiple:** A,B -> C\n4. **Hierarchical:** A->B, A->C\n5. **Hybrid:** Combination."}
            ]
        }
    },

    # ---------------- PYTHON UNIT 3 ----------------
    {
        "unit": "Unit 3: Plotting (Matplotlib)",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Matplotlib?", "answer": "2D plotting library for Python."},
                {"question": "What is pyplot?", "answer": "Submodule providing MATLAB-like plotting interface."},
                {"question": "Function to show plot?", "answer": "plt.show()"},
                {"question": "Function to save plot?", "answer": "plt.savefig('name.png')"},
                {"question": "Function to title plot?", "answer": "plt.title('Text')"},
                {"question": "What is X-label?", "answer": "Label for the horizontal axis."},
                {"question": "What is Y-label?", "answer": "Label for the vertical axis."},
                {"question": "What is a Legend?", "answer": "Key describing elements of the graph."},
                {"question": "What is a Grid?", "answer": "Lines intersecting the graph to aid readability."},
                {"question": "Define Scatter Plot.", "answer": "Graph using dots to represent values for two variables."},
                {"question": "Define Bar Plot.", "answer": "Graph using bars to compare categories."},
                {"question": "Define Histogram.", "answer": "Graph showing frequency distribution of data."},
                {"question": "Define Pie Chart.", "answer": "Circular chart divided into sectors."},
                {"question": "What is marker?", "answer": "Symbol used to represent points (e.g., 'o', 'x')."},
                {"question": "What is subplot?", "answer": "Method to display multiple distinct plots in one figure."}
            ],
            "Part B (2-Marks)": [
                {"question": "Steps to create a plot.", "answer": "1. Import library (import matplotlib.pyplot as plt)\n2. Define Data\n3. Plot data (plt.plot)\n4. Show plot (plt.show)"},
                {"question": "Line Plot vs Scatter Plot.", "answer": "• Line: Connects points (Trends).\n• Scatter: Individual points (Correlation)."},
                {"question": "Bar Plot vs Histogram.", "answer": "• Bar: Categorical data (Gaps between bars).\n• Histogram: Continuous freq data (No gaps)."},
                {"question": "Explain subplot() syntax.", "answer": "• plt.subplot(nrows, ncols, index)\n• e.g. (2, 2, 1) creates 2x2 grid, selects 1st spot."},
                {"question": "What is Figure and Axes?", "answer": "• Figure: The whole window/page.\n• Axes: The plot region (coordinates) inside the figure."},
                {"question": "How to change Line Style?", "answer": "• color='red' (or 'r')\n• linestyle='dashed' (or '--')\n• linewidth=2"},
                {"question": "How to add Legend?", "answer": "• Add label to plot: plt.plot(x, y, label='Data')\n• Call plt.legend()"},
                {"question": "Exploding a Pie Chart.", "answer": "• Pulling a slice out for emphasis.\n• usage: pie(data, explode=[0, 0.1, 0])"},
                {"question": "What is a Box Plot?", "answer": "• Visualizes statistical summary (Min, Q1, Median, Q3, Max).\n• Shows outliers."},
                {"question": "What is DPI?", "answer": "• Dots Per Inch.\n• Determines quality/resolution of saved image."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Matplotlib Architecture.", "answer": "1. **Backend Layer:** Drawing area (Renderer).\n2. **Artist Layer:** Things you see (Lines, Text).\n3. **Scripting Layer:** Pyplot (User interface)."},
                {"question": "Code to plot y = x^2.", "answer": "import matplotlib.pyplot as plt\nx = [1,2,3]\ny = [1,4,9]\nplt.plot(x,y)\nplt.show()"},
                {"question": "Explain Histogram with example.", "answer": "1. **Task:** Show age distribution.\n2. **Bins:** Ranges (0-10, 10-20).\n3. **Function:** plt.hist(ages, bins=5)."}
            ],
            "Part D (5-Marks)": [
                {"question": "Discuss various types of Plots and their uses.", "answer": "1. **Line:** trends over time.\n2. **Bar:** Comparing categories.\n3. **Scatter:** Relationship between X and Y.\n4. **Pie:** Part-to-whole relationship.\n5. **Hist:** Distribution.\n6. **Box:** Outliers/Summary."},
                {"question": "Detailed customization of a plot.", "answer": "Explain Titles, Labels, Ticks, Legend, Colors, Markers, Linestyles, Grid, and Saving."}
            ]
        }
    },

    # ---------------- PYTHON UNIT 4 ----------------
    {
        "unit": "Unit 4: Network & GUI",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Tkinter?", "answer": "Standard Python Interface for GUI (Tk)."},
                {"question": "What is GUI?", "answer": "Graphical User Interface."},
                {"question": "What consists of a GUI?", "answer": "Windows, Widgets (Buttons, Labels), and Events."},
                {"question": "What is a mainloop?", "answer": "Infinite loop that waits for and processes events."},
                {"question": "Define Widget.", "answer": "Controls on a GUI window (Button, Textbox)."},
                {"question": "What is a Socket?", "answer": "Endpoint for sending/receiving data across a network."},
                {"question": "What is IP Address?", "answer": "Unique label assigned to devices on a network."},
                {"question": "What is Port Number?", "answer": "Numeric identifier for a specific process/service."},
                {"question": "Define TCP.", "answer": "Transmission Control Protocol (Reliable, Connection-oriented)."},
                {"question": "Define UDP.", "answer": "User Datagram Protocol (Unreliable, Connection-less)."},
                {"question": "What is a Server?", "answer": "Device/Program providing resources/services."},
                {"question": "What is a Client?", "answer": "Device/Program requesting resources."},
                {"question": "What is a Layout Manager?", "answer": "Mechanism to arrange widgets in a window."},
                {"question": "Function to bind event?", "answer": "widget.bind(event, function)."},
                {"question": "What is Entry widget?", "answer": "Widget for single-line text input."}
            ],
            "Part B (2-Marks)": [
                {"question": "Explain Pack Layout.", "answer": "• Arranges widgets in blocks (Top, Bottom, Left, Right).\n• Simple but limited control."},
                {"question": "Explain Grid Layout.", "answer": "• Arranges widgets in rows and columns.\n• Table-like structure. Very flexible."},
                {"question": "Explain Place Layout.", "answer": "• Fixed positioning (x=50, y=100).\n• Precise but not responsive."},
                {"question": "Essential functions of Socket.", "answer": "• bind(): Link to port.\n• listen(): Wait for connections.\n• accept(): Accept connection.\n• connect(): Connect to server."},
                {"question": "TCP vs UDP.", "answer": "• TCP: Reliable, Slow, Ordered (Email).\n• UDP: Unreliable, Fast, Unordered (Video Stream)."},
                {"question": "Client-Server Model.", "answer": "• Distributed architecture.\n• Server waits. Client initiates request. Server responds."},
                {"question": "What is an Event?", "answer": "• Action causing something to happen.\n• e.g., Mouse Click, Key Press."},
                {"question": "What is Canvas?", "answer": "• Widget for drawing shapes (Lines, Ovals).\n• Used for custom graphics."},
                {"question": "Difference between Label and Entry.", "answer": "• Label: Displays text (Read-Only).\n• Entry: Accepts text (Input)."},
                {"question": "What is localhost?", "answer": "• Hostname '127.0.0.1'.\n• Refers to the current computer."}
            ],
            "Part C (3-Marks)": [
                {"question": "Steps to create a GUI App.", "answer": "1. Import tkinter.\n2. Create main window (Tk()).\n3. Add widgets (Button, Label).\n4. Choose Layout (pack).\n5. Enter Main Loop."},
                {"question": "Steps in UDP Communication.", "answer": "1. Create Socket (SOCK_DGRAM).\n2. sendto(msg, addr) - No connection needed.\n3. recvfrom(size)."},
                {"question": "Explain Button Widget logic.", "answer": "1. **Visual:** Clickable area.\n2. **command:** Parameter linking to a function.\n3. **Usage:** b = Button(root, text='Ok', command=my_func)."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain Widgets in Tkinter (List 5).", "answer": "1. **Label:** Text.\n2. **Button:** Action.\n3. **Entry:** Input.\n4. **Checkbutton:** Toggle.\n5. **Radiobutton:** Selection.\n6. **Canvas:** Drawing."},
                {"question": "Explain Connection-Oriented vs Connection-less.", "answer": "Detailed comparison of TCP (Phone call) vs UDP (Mail). Reliability vs Speed."}
            ]
        }
    },

    # ---------------- PYTHON UNIT 5 ----------------
    {
        "unit": "Unit 5: Database (MySQL)",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is DBMS?", "answer": "Database Management System."},
                {"question": "What is RDBMS?", "answer": "Relational DBMS (saves data in tables)."},
                {"question": "What is SQL?", "answer": "Structured Query Language."},
                {"question": "What is a Table?", "answer": "Collection of related data consisting of columns and rows."},
                {"question": "What is Primary Key?", "answer": "Column uniquely identifying each record."},
                {"question": "What is Foreign Key?", "answer": "Key linking two tables."},
                {"question": "What is MySQL Connector?", "answer": "Driver enabling Python app to talk to MySQL."},
                {"question": "What is a Cursor?", "answer": "Control structure used to traverse and fetch records."},
                {"question": "What is execute()?", "answer": "Cursor method to run SQL command."},
                {"question": "What is commit()?", "answer": "Method to save changes permanently."},
                {"question": "What is rollback()?", "answer": "Method to undo unsaved changes."},
                {"question": "What is fetchone()?", "answer": "Retrieves the next row of a query result."},
                {"question": "What is fetchall()?", "answer": "Retrieves all rows of a query result."},
                {"question": "What is DDL?", "answer": "Data Definition Language (CREATE, DROP)."},
                {"question": "What is DML?", "answer": "Data Manipulation Language (INSERT, UPDATE)."}
            ],
            "Part B (2-Marks)": [
                {"question": "Steps to connect Python to DB.", "answer": "1. Import connector.\n2. Establish connection object.\n3. Create Cursor object.\n4. Execute Query."},
                {"question": "DDL vs DML.", "answer": "• DDL: Structure (Create Table).\n• DML: Data (Insert into Table)."},
                {"question": "Why use try-except in DB code?", "answer": "• To handle connection errors gracefully.\n• To ensure connection is closed (finally block)."},
                {"question": "Parameterized Queries.", "answer": "• Using %s placeholders instead of formatting.\n• Prevents SQL Injection attacks."},
                {"question": "What is SQL Injection?", "answer": "• Malicious code inserted into strings.\n• Can destroy database."},
                {"question": "Difference between NULL and 0.", "answer": "• NULL: Missing value (Unknown).\n• 0: A numerical value."},
                {"question": "What is a Query?", "answer": "• A request for data or action (SELECT, INSERT)."},
                {"question": "Close() method importance.", "answer": "• Frees up resources/connnections.\n• Prevents memory leaks."},
                {"question": "What is Rowcount?", "answer": "• Property returning number of rows affected by query."},
                {"question": "Host, User, Password.", "answer": "• Credentials required for connection string."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain CRUD.", "answer": "1. **C**reate: INSERT.\n2. **R**ead: SELECT.\n3. **U**pdate: UPDATE.\n4. **D**elete: DELETE."},
                {"question": "Explain Transaction Management.", "answer": "1. **ACID Properties.**\n2. **Commit:** Save.\n3. **Rollback:** Undo error.\n4. Critical for banking systems."},
                {"question": "Code to fetch data.", "answer": "cur.execute('SELECT * FROM Users')\nfor row in cur.fetchall():\n  print(row)"}
            ],
            "Part D (5-Marks)": [
                {"question": "Detailed Steps for Database Connectivity.", "answer": "1. Import `mysql.connector`.\n2. `conn = connect(...)`.\n3. `cur = conn.cursor()`.\n4. `cur.execute(sql)`.\n5. `conn.commit()`.\n6. `conn.close()`."},
                {"question": "Explain SQL Commands (Insert, Select, Update, Delete).", "answer": "Detailed syntax and usage of DML commands."}
            ]
        }
    },

    # ---------------- ML UNIT 1 (Ref: Unit 6) ----------------
    {
        "unit": "Unit 6: Introduction to ML",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define Machine Learning.", "answer": "ML is a field of AI where computers learn from data without being explicitly programmed."},
                {"question": "What is Artificial Intelligence?", "answer": "AI is the simulation of human intelligence processes by machines, especially computer systems."},
                {"question": "Define Deep Learning.", "answer": "A subset of ML based on artificial neural networks with multiple layers (Deep Neural Networks)."},
                {"question": "What is a 'Label' in ML?", "answer": "The target variable or the answer that the model is trying to predict."},
                {"question": "Define 'Feature'.", "answer": "An individual measurable property or characteristic of a phenomenon being observed."},
                {"question": "What is a Training Set?", "answer": "The subset of the dataset used to train or fit the model."},
                {"question": "What is a Testing Set?", "answer": "The subset of the dataset used to evaluate the trained model's performance."},
                {"question": "Define Supervised Learning.", "answer": "A type of learning where the model is trained on labeled data (Input-Output pairs)."},
                {"question": "Define Unsupervised Learning.", "answer": "Learning where the model works on unlabeled data to find hidden patterns or structures."},
                {"question": "What is Reinforcement Learning?", "answer": "A learning method where an agent learns by interacting with an environment and receiving rewards."},
                {"question": "What is a Dataset?", "answer": "A collection of data samples used for training and testing ML models."},
                {"question": "List one example of Supervised Learning.", "answer": "Spam Email Filtering (classifying email as 'Spam' or 'Not Spam')."},
                {"question": "List one example of Unsupervised Learning.", "answer": "Customer Segmentation (Grouping customers based on purchasing behavior)."},
                {"question": "What is 'Noise' in data?", "answer": "Irrelevant or meaningless data that can confuse the model."},
                {"question": "Who coined the term 'Machine Learning'?", "answer": "Arthur Samuel (1959)."}
            ],
            "Part B (2-Marks)": [
                {"question": "Differentiate AI and ML.", "answer": "• AI: Broader concept of smart machines mimicking human behavior.\n• ML: Subset of AI where machines learn from data to improve performance."},
                {"question": "State the difference between Labeled and Unlabeled data.", "answer": "• Labeled: Data comes with the correct answer/tag (e.g., Image of Cat + Label 'Cat').\n• Unlabeled: Raw data with no tags (e.g., Image of Cat with no info)."},
                {"question": "List 4 applications of Machine Learning.", "answer": "• Self-Driving Cars\n• Fraud Detection\n• Product Recommendation\n• Voice Assistants (Siri/Alexa)"},
                {"question": "What are the 3 main types of Learning?", "answer": "• Supervised Learning\n• Unsupervised Learning\n• Reinforcement Learning"},
                {"question": "Define 'Preprocessing' in ML.", "answer": "• Process of cleaning and transforming raw data.\n• Includes handling missing values, scaling, and encoding."},
                {"question": "What is the goal of the 'Training' phase?", "answer": "• To allow the model to learn patterns from the training data.\n• To minimize the error between predicted and actual results."},
                {"question": "Why do we split data into Train and Test?", "answer": "• To prevent Overfitting (memorizing data).\n• To evaluate how well the model acts on unseen (new) data."},
                {"question": "What is Sem-Supervised Learning?", "answer": "• Uses a small amount of labeled data and a large amount of unlabeled data.\n• Combines benefits of both Supervised and Unsupervised learning."},
                {"question": "Give 2 examples of Regression problems.", "answer": "• Predicting House Prices.\n• Predicting Stock Market trends."},
                {"question": "Give 2 examples of Classification problems.", "answer": "• Identifying tumors as Benign/Malignant.\n• Classifying images as Cat/Dog."},
                {"question": "What is 'Generalization'?", "answer": "• The ability of a model to adapt to new, unseen data.\n• A model with good generalization performs well on test data."},
                {"question": "What is the role of an 'Algorithm' in ML?", "answer": "• It is the mathematical logic/procedure used to find patterns in data.\n• Examples: Linear Regression, Decision Trees."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Supervised Learning with an example.", "answer": "1. **Definition:** Learning where the machine is taught using data that is 'labeled'. The model maps Input (X) to Output (Y).\n2. **Explanation:** Like a teacher–student scenario. The teacher provides questions and answers. The student learns the logic.\n3. **Example:** Face Recognition (Input: Photo, Label: Name of person)."},
                {"question": "Explain Unsupervised Learning with an example.", "answer": "1. **Definition:** Learning where the machine is given data with NO labels. It must find structure on its own.\n2. **Explanation:** Like a baby learning about objects without a teacher. It groups similar things together.\n3. **Example:** News Categorization (Grouping similar articles by topic)."},
                {"question": "Explain Reinforcement Learning with an example.", "answer": "1. **Definition:** Learning by interacting with an environment to maximize rewards/minimize penalties.\n2. **Explanation:** Trial and error method. Good actions get +Points, bad actions get -Points.\n3. **Example:** Training a robot to walk (Falls = Penalty, Steps = Reward)."},
                {"question": "Comparing AI, ML, and DL.", "answer": "1. **AI:** The big umbrella. Any machine acting smart.\n2. **ML:** A subset of AI. Machines that simple learn from data (Statistical).\n3. **DL:** A subset of ML. Machines that learn via Neural Networks (Brain-like structure)."},
                {"question": "Explain 'Data Gathering' step in ML.", "answer": "1. **Definition:** The first and most crucial step of collecting raw data.\n2. **Explanation:** Quality and quantity of data determine model success. Identify sources (DB, API, Sensors).\n3. **Example:** Scraping tweets to analyze public sentiment."},
                {"question": "Explain 'Model Evaluation' step.", "answer": "1. **Definition:** Testing the trained model on a separate 'Test Set'.\n2. **Explanation:** Checking metrics like Accuracy. If performance is poor, we retune parameters.\n3. **Example:** Checking if a Spam filter catches 99% of spam emails."},
                {"question": "What is Deep Learning effectively?", "answer": "1. **Definition:** ML technique inspired by the human brain's neural networks.\n2. **Explanation:** Uses layers of algorithms to learn features from vast amounts of data. Good for unstructured data.\n3. **Example:** ChatGPT (Text generation), Tesla Autopilot (Vision)."},
                {"question": "Difference between Regression and Classification.", "answer": "1. **Regression:** Prediction of continuous/numerical value (e.g., Temperature 34.5°C).\n2. **Classification:** Prediction of categorical/discrete label (e.g., Hot/Cold).\n3. **Connection:** Both are types of Supervised Learning."},
                {"question": "What happens during 'Prediction' phase?", "answer": "1. **Definition:** The final stage where the model is deployed for real use.\n2. **Explanation:** We feed it new, real-world live data, and it outputs the result based on training.\n3. **Example:** Netflix suggesting a movie you might like right now."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain the 7 Steps of Machine Learning in detail.", "answer": "1. **Data Gathering:** Collecting information from various sources (Files, Databases, APIs).\n2. **Data Preprocessing:** Cleaning data (removing nulls, noise) and formatting it for the machine.\n3. **Choose Model:** Selecting the right algorithm (e.g., Regression for numbers, Decision Tree for rules).\n4. **Train Model:** The learning phase. The model processes data to find patterns.\n5. **Test Model:** Evaluating performance on unseen data to ensure it works.\n6. **Parameter Tuning:** Tweaking settings (Hyperparameters) to improve accuracy.\n7. **Prediction:** Deploying the model to solve the real-world problem."},
                {"question": "Discuss Real-World Applications of Machine Learning.", "answer": "1. **Healthcare:** Diagnosing diseases; Cancer detection.\n2. **Finance:** Fraud detection; Algo trading.\n3. **E-commerce:** Recommendations (Netflix/Amazon).\n4. **Social Media:** Face tagging; Feed curation.\n5. **Transport:** Self-Driving cars (Tesla).\n6. **Significance:** Solves complex problems at scale and automates tasks."},
                {"question": "Detailed comparison of Supervised vs Unsupervised Learning.", "answer": "1. **Labeling:** Supervised uses Labeled data. Unsupervised uses Unlabeled data.\n2. **Goal:** Supervised allows prediction (Output). Unsupervised allows discovery (Structure).\n3. **Knowledge:** Supervised is 'Teacher-based'. Unsupervised is 'Self-learning'.\n4. **Algorithms:** \n   - Supervised: Linear Regression, SVM.\n   - Unsupervised: K-Means, PCA.\n5. **Evaluation:** Supervised is exact (Accuracy). Unsupervised is subjective."}
            ]
        }
    },
    
    # ---------------- ML UNIT 2 ----------------
    {
        "unit": "Unit 7: Supervised Learning",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Regression?", "answer": "Predicting a continuous numerical value."},
                {"question": "What is Classification?", "answer": "Predicting a categorical class label."},
                {"question": "What is a Hyperplane?", "answer": "The decision boundary that separates different classes."},
                {"question": "What are Support Vectors?", "answer": "Data points closest to the hyperplane influencing its position."},
                {"question": "Define Confusion Matrix.", "answer": "A table describing the performance of a classification model."},
                {"question": "What is True Positive?", "answer": "Correctly predicted Positive tuple."},
                {"question": "What is False Positive?", "answer": "Incorrectly predicted Positive tuple (Type I Error)."},
                {"question": "What is Accuracy?", "answer": "Percentage of correct predictions over total predictions."},
                {"question": "Define Overfitting.", "answer": "Model learns noise and detail of training data too well, failing on test data."},
                {"question": "Define Underfitting.", "answer": "Model is too simple to capture the pattern in data."},
                {"question": "What is Naive Bayes?", "answer": "Classifier based on Bayes Theorem assuming feature independence."},
                {"question": "What is Logistic Regression?", "answer": "Regression model used for Binary Classification (Probability)."},
                {"question": "What is Scaling?", "answer": "Modifying features to lie within a specific range (e.g. 0-1)."},
                {"question": "What is Binarization?", "answer": "Converting numerical features into binary (0/1)."},
                {"question": "What is K-Fold Cross Validation?", "answer": "Technique to split data into K subsets for testing stability."}
            ],
            "Part B (2-Marks)": [
                {"question": "Linear vs Logistic Regression.", "answer": "• Linear: Output is continuous value (Price).\n• Logistic: Output is Probability/Class (0 or 1)."},
                {"question": "Explain Kernel Trick.", "answer": "• Transforming data into higher dimension.\n• Allows linear separation of non-linear data."},
                {"question": "What is Standardization?", "answer": "• Rescaling data to have Mean=0 and StdDev=1.\n• Helps algorithms like SVM and K-Means converge."},
                {"question": "What is Label Encoding?", "answer": "• Converting text labels into numbers.\n• e.g. 'Red'->0, 'Blue'->1, 'Green'->2."},
                {"question": "Precision vs Recall.", "answer": "• Precision: Accuracy of positive predictions (Quality).\n• Recall: Ability to find all positives (Quantity)."},
                {"question": "Describe SVM.", "answer": "• Support Vector Machine.\n• Finds hyperplane maximizing the margin between classes."},
                {"question": "Why use Cross-Validation?", "answer": "• To check if model is stable.\n• Prevents bias from a single train/test split."},
                {"question": "What is F1-Score?", "answer": "• Harmonic mean of Precision and Recall.\n• Best metric for imbalanced datasets."},
                {"question": "What is Correlation?", "answer": "• Statistical relationship between two variables.\n• Positive (Both rise), Negative (One rises, one falls)."},
                {"question": "Difference between Training and Testing.", "answer": "• Training: Model sees answers, learns patterns.\n• Testing: Model hides answers, we verify patterns."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Linear Regression Equation.", "answer": "1. **y = mx + c**\n2. **y:** Dependent variable (Target).\n3. **x:** Independent variable (Feature).\n4. **m:** Slope (Weight).\n5. **c:** Intercept (Bias)."},
                {"question": "Explain Confusion Matrix terms.", "answer": "1. **TP:** Correctly identified Positive.\n2. **TN:** Correctly identified Negative.\n3. **FP:** Incorrectly identified Positive (Type I).\n4. **FN:** Incorrectly identified Negative (Type II)."},
                {"question": "Explain Naive Bayes assumption.", "answer": "1. **Assumes Independence:** Presence of one feature is unrelated to any other.\n2. **Example:** Fruit is Red, Round, Apple. Red doesn't depend on Round.\n3. **Result:** Very fast training."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain Support Vector Machines (SVM) in detail.", "answer": "1. **Goal:** Find optimal hyperplane.\n2. **Margin:** Gap between classes. SVM maximizes this.\n3. **Support Vectors:** Critical points defining the margin.\n4. **Kernel Trick:** Handling non-linear data.\n5. **Usage:** Classification and Regression."},
                {"question": "Explain Classification Performance Metrics.", "answer": "1. **Accuracy:** (TP+TN)/Total.\n2. **Precision:** TP/(TP+FP).\n3. **Recall:** TP/(TP+FN).\n4. **F1:** 2*P*R/(P+R).\n5. **Usage:** Accuracy for balanced, F1 for imbalanced."}
            ]
        }
    },

    # ---------------- ML UNIT 3 ----------------
    {
        "unit": "Unit 8: Unsupervised Learning",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define Clustering.", "answer": "Grouping sets of objects so that objects in the same group are similar."},
                {"question": "What is K-Means?", "answer": "Iterative, centroid-based clustering algorithm."},
                {"question": "What is a Centroid?", "answer": "The center point of a cluster."},
                {"question": "What is Unlabeled Data?", "answer": "Data without target labels."},
                {"question": "Define Dimensionality Reduction.", "answer": "Reducing the number of random variables (features) under consideration."},
                {"question": "What is a Dendrogram?", "answer": "Tree diagram showing taxonomic relationships in hierarchical clustering."},
                {"question": "What is Vector Quantization?", "answer": "Compression technique using clustering (e.g. reducing colors)."},
                {"question": "Define Mean Shift.", "answer": "Non-parametric clustering technique using sliding windows."},
                {"question": "What is Inertia?", "answer": "Sum of squared distances of samples to their closest cluster center."},
                {"question": "What is Euclidean Distance?", "answer": "Straight line distance between two points."},
                {"question": "What is Manhattan Distance?", "answer": "Sum of absolute differences of coordinates."},
                {"question": "What is Agglomerative Clustering?", "answer": "Bottom-up hierarchical clustering (merging)."},
                {"question": "What is Divisive Clustering?", "answer": "Top-down hierarchical clustering (splitting)."},
                {"question": "What is Anomaly Detection?", "answer": "Identifying rare items that differ significantly from majority."},
                {"question": "What is K in K-Means?", "answer": "The number of clusters specified by the user."}
            ],
            "Part B (2-Marks)": [
                {"question": "Supervised vs Unsupervised.", "answer": "• Supervised: Labeled data, Prediction (Regression/Class).\n• Unsupervised: Unlabeled data, Discovery (Clustering)."},
                {"question": "How to choose K in K-Means?", "answer": "• Elbow Method: Plot Inertia vs K.\n• Look for the 'elbow' point where improvement slows."},
                {"question": "K-Means vs Hierarchical.", "answer": "• K-Means: Faster, requires K, assumes spherical.\n• Hierarchical: Slower, builds tree, no K needed initially."},
                {"question": "What is Semi-Supervised Learning?", "answer": "• Hybrid approach.\n• Train on small labeled set -> Label the rest -> Train on all."},
                {"question": "Applications of Clustering.", "answer": "• Customer Segmentation (Marketing).\n• Image Compression.\n• Document Classification."},
                {"question": "What is the curse of dimensionality?", "answer": "• As dimensions increase, data becomes sparse.\n• Distance metrics lose meaning."},
                {"question": "Explain Agglomerative approach.", "answer": "• Start: Each point is a cluster.\n• Step: Merge 2 closest clusters.\n• End: One giant cluster."},
                {"question": "What is Mean Shift used for?", "answer": "• Image Segmentation.\n• Tracking objects in video."},
                {"question": "Limitations of K-Means.", "answer": "• Must pick K.\n• Sensitive to outliers.\n• Only finds spherical clusters."},
                {"question": "What is Silhouette Score?", "answer": "• Metric (-1 to 1) measuring how similar object is to its own cluster vs others."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain K-Means Algorithm steps.", "answer": "1. **Initialize:** Pick K random centroids.\n2. **Assign:** Assign each point to nearest centroid.\n3. **Update:** Move centroid to mean of assigned points.\n4. **Repeat:** Until centroids stop moving."},
                {"question": "Explain Image Compression with Clustering.", "answer": "1. **Concept:** Vector Quantization.\n2. **Process:** Cluster miilions of colors into K=64 colors.\n3. **Replace:** Replace every pixel with nearest centroid color.\n4. **Result:** Smaller file size."},
                {"question": "Explain Hierarchical Clustering types.", "answer": "1. **Agglomerative:** Bottom-Up. Merge closest pairs.\n2. **Divisive:** Top-Down. Split one cluster recursively.\n3. **Visualization:** Dendrogram."}
            ],
            "Part D (5-Marks)": [
                {"question": "Detailed explanation of K-Means Clustering.", "answer": "1. **Concept:** Partition N obs into K sets.\n2. **Objective:** Minimize WCSS (With-Cluster Sum of Squares).\n3. **Algorithm:** Init -> Assignment -> Update -> Convergence.\n4. **Pros:** Fast, Simple.\n5. **Cons:** Local global, K selection."},
                {"question": "Compare K-Means, Hierarchical, and DBSCAN.", "answer": "1. **K-Means:** Centroid-based, spherical clusters, fast.\n2. **Hierarchical:** Tree-based, nested clusters, slow, easy viz.\n3. **DBSCAN:** Density-based, arbitrary shapes, handles noise."}
            ]
        }
    },

    # ---------------- ML UNIT 4 ----------------
    {
        "unit": "Unit 9: Natural Language Processing",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is NLP?", "answer": "Interaction between computers and human language."},
                {"question": "Define Tokenization.", "answer": "Breaking text into words or sentences."},
                {"question": "What is a Corpus?", "answer": "A large collection of text documents."},
                {"question": "What is a Stopword?", "answer": "Common word (the, is) filtered out during processing."},
                {"question": "Define Stemming.", "answer": "Reducing words to their root form by chopping suffixes."},
                {"question": "Define Lemmatization.", "answer": "Reducing words to their dictionary root form."},
                {"question": "What is BoW?", "answer": "Bag of Words model (Frequency counting)."},
                {"question": "What is TF-IDF?", "answer": "Term Frequency-Inverse Document Frequency."},
                {"question": "What is Sentiment Analysis?", "answer": "Identifying emotional tone (Pos/Neg) in text."},
                {"question": "What is Chunking?", "answer": "Grouping tokens into meaningful phrases (Noun Phrases)."},
                {"question": "What is NLTK?", "answer": "Natural Language Toolkit library in Python."},
                {"question": "What is a Document?", "answer": "A single unit of text (tweet, email, book)."},
                {"question": "What is Vocabulary?", "answer": "Set of unique words in the corpus."},
                {"question": "What is POS Tagging?", "answer": "Part-of-Speech tagging (identifying Noun, Verb, Adj)."},
                {"question": "What is a Chatbot?", "answer": "Software simulating human conversation."}
            ],
            "Part B (2-Marks)": [
                {"question": "Stemming vs Lemmatization.", "answer": "• Stemming: Heuristic chopping (Fast, Crude). 'Caring'->'Car'.\n• Lemmatization: Morphological analysis (Slow, Accurate). 'Caring'->'Care'."},
                {"question": "Explain Bag of Words.", "answer": "• Representing text as numerical features.\n• Discards grammar/order, keeps distinct word counts."},
                {"question": "Explain TF-IDF.", "answer": "• TF: How often word appears in doc.\n• IDF: How rare word is in corpus.\n• Highlights unique/important words."},
                {"question": "Applications of NLP.", "answer": "• Spam Detection\n• Machine Translation (Google Translate)\n• Voice Assistants\n• Sentiment Analysis"},
                {"question": "What is Named Entity Recognition (NER)?", "answer": "• Identifying entities in text.\n• Persons, Organizations, Locations, Dates."},
                {"question": "Why remove Stopwords?", "answer": "• They add noise and dimension without adding meaning.\n• Removing 'the', 'and' improves efficiency."},
                {"question": "What is Text Cleaning?", "answer": "• Lowercasing, Removing Punctuation, Removing HTML tags.\n• Preparing raw text for ML."},
                {"question": "What is Word Embedding?", "answer": "• Representing words as dense vectors.\n• Captures semantic meaning (King - Man + Woman = Queen)."},
                {"question": "Challenges in NLP.", "answer": "• Ambiguity (Sarcasm).\n• Slang/Idioms.\n• Multiple Languages."},
                {"question": "What is a Bi-gram?", "answer": "• Sequence of 2 adjacent elements.\n• e.g., 'New York', 'ice cream'."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Text Preprocessing Pipeline.", "answer": "1. **Clean:** Lowercase, remove noise.\n2. **Tokenize:** Split into words.\n3. **Normalize:** Remove stopwords, Stem/Lemmatize.\n4. **Vectorize:** Convert to numbers."},
                {"question": "Explain Sentiment Analysis logic.", "answer": "1. **Input:** \"I love this movie!\"\n2. **Feature:** Love (Positive weight).\n3. **Model:** Logistic Regression.\n4. **Output:** Positive Class."},
                {"question": "Explain Chunking.", "answer": "1. **Goal:** Extract phrases.\n2. **Input:** POS Tagged text.\n3. **Grammar:** NP: {<DT>?<JJ>*<NN>}.\n4. **Output:** Tree structure."}
            ],
            "Part D (5-Marks)": [
                {"question": "Detailed explanation of Vectorization (BoW vs TF-IDF).", "answer": "1. **BoW:** Simple counting. Problem: Frequent words dominate.\n2. **TF-IDF:** Penalizes frequent words. Enhances rare, distinct words.\n3. **Result:** Better feature set for classification."},
                {"question": "Case Study: Spam Filtering.", "answer": "1. **Data:** Collection of Emails (Ham/Spam).\n2. **Process:** Clean text, Tokenize, Vectorize (TF-IDF).\n3. **Train:** Naive Bayes Classifier.\n4. **Test:** Check accuracy.\n5. **Predict:** New email -> Spam/Not Spam."}
            ]
        }
    },

    # ---------------- ML UNIT 5 ----------------
    {
        "unit": "Unit 10: Computer Vision",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Computer Vision?", "answer": "Field enabling computers to 'see' and understand images."},
                {"question": "What is typical image size?", "answer": "Width x Height x Channels (e.g. 640x480x3)."},
                {"question": "What is a Pixel?", "answer": "Picture Element. Smallest unit of an image."},
                {"question": "What is OpenCV?", "answer": "Open Source Computer Vision Library."},
                {"question": "What is RGB?", "answer": "Red Green Blue color model."},
                {"question": "What is Grayscale?", "answer": "Image with only intensity values (0=Black, 255=White)."},
                {"question": "What is Thresholding?", "answer": "Converting grayscale image to binary (Black/White)."},
                {"question": "What is a Haar Cascade?", "answer": "ML-based object detection method."},
                {"question": "Function to read image?", "answer": "cv2.imread()"},
                {"question": "Function to show image?", "answer": "cv2.imshow()"},
                {"question": "What is FPS?", "answer": "Frames Per Second."},
                {"question": "What is Edge Detection?", "answer": "Identifying points where brightness changes sharply."},
                {"question": "What is Object Tracking?", "answer": "Locating a moving object over time."},
                {"question": "What is a ROI?", "answer": "Region of Interest (Part of image we care about)."},
                {"question": "What is Noise in image?", "answer": "Random variation of brightness/color (Grainy)."}
            ],
            "Part B (2-Marks)": [
                {"question": "How do computers see images?", "answer": "• As a matrix of numbers.\n• 0-255 representing brightness of each pixel."},
                {"question": "Grayscale vs Color.", "answer": "• Grayscale: 1 Channel (Intensity). Fast.\n• Color: 3 Channels (BGR). Slow, more info."},
                {"question": "Detection vs Tracking.", "answer": "• Detection: Search whole image every frame.\n• Tracking: Search near last known location. Faster."},
                {"question": "What are Haar Features?", "answer": "• rectangular filters (Edges, Lines).\n• Used to detect facial features (Eyes, Nose)."},
                {"question": "Explain Viola-Jones Algorithm.", "answer": "• Used for fast face detection.\n• Uses Haar Features + Integral Image + Cascade."},
                {"question": "Why use Grayscale for detection?", "answer": "• Reduces data by 66% (1 channel vs 3).\n• Most shapes/features don't depend on color."},
                {"question": "What is Canny Edge Detection?", "answer": "• Popular Algorithm to find edges.\n• Steps: Noise reduction -> Gradient calc -> Thresholding."},
                {"question": "Applications of CV.", "answer": "• Face Unlock (Phones).\n• License Plate Recognition.\n• Medical Imaging (X-Ray)."},
                {"question": "Reading Video in OpenCV.", "answer": "• Use cv2.VideoCapture(0).\n• Read frame-by-frame in a loop."},
                {"question": "WaitKey function.", "answer": "• cv2.waitKey(1)\n• Pauses for x milliseconds. Used to capture keyboard input."}
            ],
            "Part C (3-Marks)": [
                {"question": "Explain Face Detection steps.", "answer": "1. **Load:** CascadeClassifier('face.xml').\n2. **Read:** Image to Gray.\n3. **Detect:** face_cascade.detectMultiScale(gray).\n4. **Draw:** Rectangle around faces."},
                {"question": "Explain Drowsiness Detection concept.", "answer": "1. **Detect:** Face then Eyes.\n2. **Measure:** Eye Aspect Ratio (Openness).\n3. **Logic:** If EAR < Threshold for Time T -> Alert."},
                {"question": "Object Detection Pipeline.", "answer": "1. **Input:** Frame.\n2. **Preprocess:** Resize/Gray.\n3. **Feature Extract:** Find patterns.\n4. **Classify:** Object vs Background."}
            ],
            "Part D (5-Marks)": [
                {"question": "Explain the Viola-Jones Framework.", "answer": "1. **Haar Features:** Simple rectangles to define features.\n2. **Integral Image:** Fast calculation of pixel sums.\n3. **Adaboost:** Selecting best features.\n4. **Cascading:** Rejecting negative regions instantly (Fail fast)."},
                {"question": "Real-time Object Tracking.", "answer": "Discussion of using Webcam, capturing frames, applying detection/tracking logic, and displaying output in real-time."}
            ]
        }
    }
]

def populate_expert_bank():
    with open(EXAM_FILE, 'w') as f:
        json.dump(ALL_EXPERT_DATA, f, indent=4)
    print(f"Successfully generated EXPERT QUESTION BANK for {len(ALL_EXPERT_DATA)} Units.")

if __name__ == "__main__":
    populate_expert_bank()
