import json
import random
import sys

# Configuration
UNITS = [
    "Unit 1: Python Basics",
    "Unit 2: OOP Using Python",
    "Unit 3: Plotting & Algorithms",
    "Unit 4: Network & GUI",
    "Unit 5: Database Connectivity",
    "Unit 1: Introduction to ML",
    "Unit 2: Supervised Learning",
    "Unit 3: Unsupervised Learning",
    "Unit 4: NLP",
    "Unit 5: Computer Vision"
]

# High-Quality Specific Content Database (Expanded)
# STRICT FORMATTING: \n for newlines, no markdown.
DB = {
    # --- Unit 1: Python Basics (Expanded) ---
    "Variable": {
        "def": "A variable is a symbolic name associated with a value in memory.",
        "part_b": ["It allows you to store data values for later processing.", "It improves code readability by giving meaningful names."],
        "use": "Storing data for later use.",
        "expl": "Think of a variable like a labeled box. It works by taking a value (Input), storing it in a specific memory location (Processing), and allowing you to retrieve it by name later (Output). This is crucial because it lets programs remember information like scores or names while they run.",
        "ex": "score = 100\nprint(score) # Output: 100",
        "app": ["Tracking Game Scores (Real-life).", "Storing User Input (Technical).", "Scientific Calculations (Academic)."],
        "adv": ["Easy to name and use.", "Allows data to be reused.", "Makes code readable."]
    },
    "List": {
        "def": "A list is a mutable, ordered sequence of elements.",
        "part_b": ["It is used to store multiple items in a single variable.", "It provides built-in methods to sort, search, and modify the collection."],
        "use": "Storing multiple items in one place.",
        "expl": "Think of a list like a music playlist. It works by taking multiple items (Input), organizing them in a specific order with index numbers (Processing), and letting you access or change any item instantly (Output). This makes it perfect for managing collections of things that need to stay in order.",
        "ex": "fruits = ['apple', 'banana']\nprint(fruits[0]) # Output: apple",
        "app": ["Managing a Music Playlist (Real-life).", "Storing a Class Roster (Academic).", "Processing Sensor Data Logs (Technical)."],
        "adv": ["Can store mixed data types.", "Easy to add/remove items (Dynamic).", "Built-in sorting and searching."]
    },
    "Tuple": {
        "def": "A tuple is an immutable, ordered sequence of elements.",
        "part_b": ["It stores fixed collections of items that cannot be changed.", "It can be used as a key in dictionaries due to immutability."],
        "use": "Storing fixed data.",
        "expl": "Think of a tuple like a sealed envelope. It works by accepting data once (Input), locking it permanently in memory (Processing), and allowing read-only access thereafter (Output). It is essential for data that should never change during the program.",
        "ex": "coords = (10, 20)\nprint(coords[0]) # Output: 10",
        "app": ["Storing Map Coordinates (Real-life).", "Configuration Settings (Technical).", "Returning Multiple Values (Academic)."],
        "adv": ["Faster than lists.", "Data remains safe from changes.", "Can be used as dictionary keys."]
    },
    "Dictionary": {
        "def": "A dictionary is an unordered collection of unique key-value pairs.",
        "part_b": ["It allows for very fast data lookup using unique keys.", "It matches real-world data structures well, like JSON."],
        "use": "Fast data retrieval via keys.",
        "expl": "Think of a dictionary like a real phone book. It works by taking a unique key, like a name (Input), hashing it to find its location (Processing), and instantly returning the associated value, like a number (Output). This structure allows for lightning-fast lookups without searching through everything.",
        "ex": "user = {'name': 'Alice', 'age': 20}\nprint(user['name']) # Output: Alice",
        "app": ["Phone Contacts App (Real-life).", "User Profiles Database (Technical).", "Product Catalog SKU lookup (Business)."],
        "adv": ["Extremely fast lookups.", "Flexible key-value structure.", "Easy to read and manage."]
    },
    "Set": {
        "def": "A set is an unordered collection of unique elements.",
        "part_b": ["It automatically removes duplicate values from a collection.", "It supports mathematical operations like union and intersection."],
        "use": "Storing unique items.",
        "expl": "Think of a set like a specific club membership list. It works by taking a group of items (Input), filtering out any duplicates automatically (Processing), and storing only unique members (Output). It is perfect for when you need to ensure no item appears twice.",
        "ex": "ids = {101, 102, 101}\nprint(ids) # Output: {101, 102}",
        "app": ["Removing Duplicate Emails (Real-life).", "Venn Diagram Logic (Academic).", "Unique Visitor Tracking (Technical)."],
        "adv": ["Guarantees uniqueness.", "Fast membership testing.", "Example of mathematical sets."]
    },
    "Function": {
        "def": "A function is a reusable block of code designed to perform a specific task.",
        "part_b": ["It promotes code reuse by avoiding repetition.", "It makes complex programs modular and easier to debug."],
        "use": "Reusable code logic.",
        "expl": "Think of a function like a kitchen blender. It works by taking raw ingredients as arguments (Input), following a set of internal instructions to process them (Processing), and delivering a finished result (Output). You can use this machine over and over without rebuilding it every time.",
        "ex": "def add(a, b):\n    return a + b\nprint(add(5, 3)) # Output: 8",
        "app": ["Automating Daily Tasks (Real-life).", "Game Character Actions like Jump (Technical).", "Calculators/Math Operations (Academic)."],
        "adv": ["Reduces Code Repetition.", "Makes Debugging Easier.", "Improves Readability."]
    },
    "Module": {
        "def": "A module is a file containing Python definitions and statements.",
        "part_b": ["It allows you to organize code into separate, manageable files.", "It enables code sharing across different projects via imports."],
        "use": "Organizing code.",
        "expl": "Think of a module like a tool chest. It works by collecting related tools/functions in one file (Input), organizing them into a namespace (Processing), and letting you 'import' specific tools when needed (Output). This keeps your main workspace clean and organized.",
        "ex": "import math\nprint(math.sqrt(16)) # Output: 4.0",
        "app": ["Using Math Formulas (Academic).", "Importing Game Assets (Technical).", "Separating Configuration (Business)."],
        "adv": ["Organizes code logically.", "Prevents name conflicts.", "Promotes reusability."]
    },
    "String": {
        "def": "A string is an immutable sequence of characters used to store text.",
        "part_b": ["It works as an array of characters that can be indexed.", "It provides methods for text manipulation like split and upper."],
        "use": "Storing text data.",
        "expl": "Think of a string like a charm bracelet. It works by taking individual characters (Input), linking them together in a specific order (Processing), and presenting them as a single text entity (Output). You can't change a bead once it's on (immutable), but you can make a new bracelet.",
        "ex": "name = 'Python'\nprint(name[0]) # Output: P",
        "app": ["Displaying Usernames (Real-life).", "Parsing File Content (Technical).", "Printing Console Messages (Business)."],
        "adv": ["Human-readable format.", "Wide range of built-in methods.", "Immutable and hashable."]
    },
    "Integer": {
        "def": "An integer is a whole number without a fractional part.",
        "part_b": ["It supports standard arithmetic operations.", "It has unlimited precision in Python 3."],
        "use": "Counting and math.",
        "expl": "Think of an integer like a counting tally. It works by taking a whole number inputs (Input), performing exact math operations (Processing), and returning a precise result (Output). It's used for things you can count, like people or apples.",
        "ex": "count = 5 + 3\nprint(count) # Output: 8",
        "app": ["Loop Counters (Technical).", "Inventory Stock (Business).", "Age Calculation (Real-life)."],
        "adv": ["Exact precision.", "Fast arithmetic.", "Simple memory usage."]
    },
    "Loop": {
        "def": "A loop is a control structure that repeats a block of code.",
        "part_b": ["It automates repetitive tasks efficiently.", "It can iterate over sequences like lists or strings."],
        "use": "Repeating tasks.",
        "expl": "Think of a loop like a music track on repeat. It works by taking a set of instructions (Input), running them over and over until a stop condition is met (Processing), and finishing when the task is done (Output). It saves you from writing the same code 100 times.",
        "ex": "for i in range(3):\n    print(i)",
        "app": ["Sending Bulk Emails (Business).", "Processing Image Pixels (Technical).", "Game Main Loop (Real-life)."],
        "adv": ["Reduces code redundancy.", "Automates heavy tasks.", "Handles dynamic data sizes."]
    },
    "Operator": {
        "def": "An operator is a symbol that performs computations on values.",
        "part_b": ["It includes arithmetic, comparison, and logical types.", "It returns a result based on the operands provided."],
        "use": "Performing calculations.",
        "expl": "Think of an operator like a math verb. It works by taking two values (Input), applying an action like addition or comparison (Processing), and giving you the result (Output). Example: '+' adds things, '==' compares them.",
        "ex": "result = 10 + 5\nprint(result) # Output: 15",
        "app": ["Calculating Prices (Business).", "Checking Conditions (Technical).", "Physics Simulations (Academic)."],
        "adv": ["Simple syntax.", "Essential for logic.", "Highly optimized."]
    },
    "Comment": {
        "def": "A comment is a non-executable line of text in code.",
        "part_b": ["It is ignored by the interpreter during execution.", "It explains the logic to humans reading the code."],
        "use": "Explaining code.",
        "expl": "Think of a comment like a sticky note on a document. It works by allowing you to write notes (Input) that the computer ignores completely (Processing) but helps the next person understand what's going on (Output).",
        "ex": "# This adds two numbers\nx = 1 + 1",
        "app": ["Documentation (Business).", "Debugging Notes (Technical).", "Teaching Code (Academic)."],
        "adv": ["Improves maintainability.", "Helps others understand logic.", "Can carry TODOs."]
    },

    # --- Unit 2: OOP (Expanded) ---
    "Class": {
        "def": "A class is a user-defined blueprint for creating objects.",
        "part_b": ["It serves as a template to define the structure and behavior of objects.", "It encapsulates data (attributes) and actions (methods)."],
        "use": "Defining object structure.",
        "expl": "Think of a class like a cookie cutter. It works by defining a specific shape and size (Input/Design), which is then pressed into dough (Processing) to create multiple identical cookies or 'objects' (Output). The class itself is just the tool; the objects are what you actually use.",
        "ex": "class Car:\n    brand = 'Toyota'\nmy_car = Car()\nprint(my_car.brand)",
        "app": ["Creating Enemies/NPCs in Video Games (Technical).", "Modeling Bank Accounts (Business).", "Designing GUI Buttons (Technical)."],
        "adv": ["Organizes Code Logically.", "Encapsulates Data.", "Enables Reuse via Objects."]
    },
    "Object": {
        "def": "An object is a specific instance of a class.",
        "part_b": ["It represents a concrete entity with its own unique data.", "It interact with other objects through methods."],
        "use": "Representing real entities.",
        "expl": "Think of an object like an actual car built from a factory blueprint. It works by taking the Class definition (Input), allocating memory for specific data like 'Red Color' (Processing), and existing as a usable entity that can drive (Output). While the class is the idea, the object is the reality.",
        "ex": "my_dog = Dog()\nmy_dog.bark()",
        "app": ["A Specific User in a Game (Real-life).", "A Button on a Website (Technical).", "An Individual Bank Account (Business)."],
        "adv": ["Intuitive representation.", "Holds specific state.", "Modular interaction."]
    },
    "Inheritance": {
        "def": "Inheritance is a mechanism where a child class derives attributes from a parent class.",
        "part_b": ["It promotes code reusability by letting children use parent code.", "It establishes a logical hierarchy (e.g., Dog is a type of Animal)."],
        "use": "Sharing features between classes.",
        "expl": "Think of inheritance like genetic DNA. It works by taking the traits of a Parent class (Input), passing them down automatically (Processing), so the Child class starts with all those features ready to use (Output). This saves you from re-typing the same code for shared behaviors.",
        "ex": "class Animal:\n    def speak(self): print('Hi')\nclass Dog(Animal):\n    pass\nd = Dog()\nd.speak()",
        "app": ["Game Unit Types: Soldier -> Hero (Technical).", "User Roles: Guest -> Admin (Business).", "UI Widgets: Box -> Checkbox (Technical)."],
        "adv": ["Saves Typing (Reusability).", "Logical Hierarchy.", "Easy to Update Common Logic."]
    },
    "Polymorphism": {
        "def": "Polymorphism allows different classes to be treated as instances of the same general class.",
        "part_b": ["It allows a single interface to control diverse objects.", "It simplifies code by treating different types uniformly."],
        "use": "Unified interface.",
        "expl": "Think of Polymorphism like a universal remote control. It works by sending a standard 'Power On' signal (Input), which different devices like TV or DVD player interpret in their own way (Processing), to turn on (Output). You don't need a different button for every single device.",
        "ex": "for shape in [Circle(), Square()]:\n    shape.draw()",
        "app": ["Universal Plugin Systems (Technical).", "Media Player Play Button (Real-life).", "Drawing distinct shapes (Academic)."],
        "adv": ["Flexible coding.", "Unified interface.", "Extensible systems."]
    },
    "Encapsulation": {
        "def": "Encapsulation is the bundling of data and methods into a single unit.",
        "part_b": ["It restricts direct access to some of an object's components.", "It protects the internal state of an object from unintended changes."],
        "use": "Protecting data.",
        "expl": "Think of Encapsulation like a capsule pill. It works by wrapping the bitter medicine/data (Input) inside a protective shell (Processing), ensuring it travels safely without being tampered with until needed (Output). It keeps the dangerous or sensitive parts hidden from the outside world.",
        "ex": "class Bank:\n    __balance = 0 # Private",
        "app": ["Secure Banking Systems (Business).", "Medical Records Privacy (Real-life).", "OS Kernel Protection (Technical)."],
        "adv": ["Data Security.", "Controlled Access.", "Modularity."]
    },
    "Abstraction": {
        "def": "Abstraction is the concept of hiding complex implementation details and showing only essentials.",
        "part_b": ["It reduces complexity by enforcing a clean interface.", "It allows users to use a system without knowing how it works internally."],
        "use": "Hiding complexity.",
        "expl": "Think of Abstraction like driving a car. It works by giving you a pedal and wheel (Input), hiding the complex engine combustion logic (Processing), and just making the car move (Output). You don't need to be a mechanic to drive.",
        "ex": "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): pass",
        "app": ["API Design (Technical).", "Car Dashboard (Real-life).", "Remote Controls (Business)."],
        "adv": ["Simplifies interface.", "Reduces impact of changes.", "Focuses on 'What' not 'How'."]
    },
    "Method": {
        "def": "A method is a function associated with an object.",
        "part_b": ["It defines the behavior of the object.", "It can modify the object's internal state."],
        "use": "Object actions.",
        "expl": "Think of a method like a verb for a noun. It works by being called on an object (Input), using the object's internal data (Processing), and performing an action like 'bark' or 'drive' (Output). It brings the static data of an object to life.",
        "ex": "s = 'hello'\nprint(s.upper()) # Output: HELLO",
        "app": ["String Manipulation (Technical).", "Game Character Attacks (Real-life).", "Bank Transactions (Business)."],
        "adv": ["Encapsulated behavior.", "Access to object state.", "Clear syntax."]
    },
    "Constructor": {
        "def": "A constructor is a special method used to initialize objects.",
        "part_b": ["It is automatically called when an object is created.", "It sets up the initial state of the object."],
        "use": "Initializing objects.",
        "expl": "Think of a constructor like the setup wizard on a new phone. It works by running the moment you turn the phone on (Input), installing default apps and settings (Processing), and handing you a ready-to-use device (Output).",
        "ex": "class Dog:\n    def __init__(self, name):\n        self.name = name",
        "app": ["Setting Default Configs (Technical).", "New User Registration (Business).", "Opening a File (Real-life)."],
        "adv": ["Ensures valid state.", "Automates setup.", "Standardizes object creation."]
    },
    "Attribute": {
        "def": "An attribute is a variable that belongs to an object or class.",
        "part_b": ["It stores data specific to the object.", "It can be accessed using dot notation."],
        "use": "Storing object data.",
        "expl": "Think of an attribute like the color of your eyes. It works by being a property attached to you (Input), stored as part of your identity (Processing), and visible when someone describes you (Output).",
        "ex": "p = Point()\np.x = 10",
        "app": ["User Settings (Real-life).", "Game Health Points (Technical).", "Product Prices (Business)."],
        "adv": ["Persists data.", "Specific to instance.", "Easily accessible."]
    },

    # --- Unit 3: Plotting (Expanded) ---
    "Array": {
        "def": "An array is a collection of items stored at contiguous memory locations.",
        "part_b": ["It is optimized for numerical computations.", "All elements must be of the same data type."],
        "use": "Numerical data storage.",
        "expl": "Think of an array like a row of identical lockers. It works by taking data items (Input), storing them fast side-by-side (Processing), and allowing instant access by locker number (Output).",
        "ex": "import numpy as np\narr = np.array([1, 2, 3])",
        "app": ["Scientific Computing (Academic).", "Image Processing (Technical).", "Financial Modeling (Business)."],
        "adv": ["Memory efficient.", "Fast math operations.", "Compact storage."]
    },
    "Histogram": {
        "def": "A histogram is a graph showing the frequency distribution of data.",
        "part_b": ["It groups data into bins.", "It helps visualize how data is spread out."],
        "use": "Visualizing distribution.",
        "expl": "Think of a histogram like sorting mail into slots by zip code. It works by taking many items (Input), grouping them into ranges or bins (Processing), and showing which bin has the biggest pile (Output).",
        "ex": "plt.hist(data, bins=5)",
        "app": ["Grade Distribution (Academic).", "Age Demographics (Business).", "Pixel Intensity (Technical)."],
        "adv": ["Shows skewness.", "Identifies outliers.", "Summarizes large data."]
    },
    "Scatter Plot": {
        "def": "A scatter plot acts as a graph of points to show relationship between two variables.",
        "part_b": ["It uses dots to represent values for two different numeric variables.", "It is used to observe relationships or correlations."],
        "use": "Checking correlation.",
        "expl": "Think of a scatter plot like a map of star systems. It works by taking X and Y coordinates (Input), plotting a dot for each pair (Processing), and revealing if they form a line or cluster (Output).",
        "ex": "plt.scatter(x, y)",
        "app": ["Height vs Weight (Real-life).", "Price vs Demand (Business).", "Study Time vs Grade (Academic)."],
        "adv": ["Shows correlations.", "Identifies clusters.", "Detects outliers."]
    },
    "Bar Chart": {
        "def": "A bar chart presents categorical data with rectangular bars.",
        "part_b": ["The height of the bar corresponds to the value.", "It is used for comparing different groups."],
        "use": "Comparing categories.",
        "expl": "Think of a bar chart like a lineup of people by height. It works by taking category totals (Input), drawing a bar for each (Processing), and letting you easily see who is tallest (Output).",
        "ex": "plt.bar(categories, values)",
        "app": ["Sales per Month (Business).", "Votes per Candidate (Real-life).", "Browser Market Share (Technical)."],
        "adv": ["Easy comparison.", "Clear categories.", "Visual impact."]
    },
    "Legend": {
        "def": "A legend is an area describing the elements of the graph.",
        "part_b": ["It maps colors or symbols to their meaning.", "It helps the viewer interpret the plot."],
        "use": "Explaining plot colors.",
        "expl": "Think of a legend like a map key. It works by taking the colors used in a chart (Input), listing what each color represents (Processing), and telling the user 'Blue means Water' (Output).",
        "ex": "plt.legend(['Sales', 'Profit'])",
        "app": ["Complex Maps (Real-life).", "Multi-line Graphs (Technical).", "Pie Charts (Business)."],
        "adv": ["Clarifies context.", "Essential for multi-series.", "Professional look."]
    },

    # --- Unit 4: Network/GUI (Expanded) ---
    "Socket": {
        "def": "A socket is an endpoint for sending or receiving data across a network.",
        "part_b": ["It allows processes on different machines to communicate.", "It uses IP address and Port number."],
        "use": "Network communication.",
        "expl": "Think of a socket like a telephone jack. It works by plugging into the network wall (Input), establishing a line to another phone (Processing), and letting voice data flow back and forth (Output).",
        "ex": "s = socket.socket()",
        "app": ["Chat Applications (Real-life).", "Web Browsers (Technical).", "Online Gaming (Business)."],
        "adv": ["Low-level control.", "Universal standard.", "Bidirectional."]
    },
    "Port": {
        "def": "A port is a number that identifies a specific process on a computer.",
        "part_b": ["It helps the OS send network data to the right application.", "Common ports include 80 for Web and 25 for Email."],
        "use": "Identifying apps.",
        "expl": "Think of a port like an apartment number. It works by taking incoming mail for the building (Input), checking the unit number (Processing), and delivering it to the specific family inside (Output). IP is the building, Port is the unit.",
        "ex": "s.bind(('localhost', 8080))",
        "app": ["Web Servers (Technical).", "Email Services (Business).", "Database Access (Real-life)."],
        "adv": ["Multiplexing.", "Organization.", "Security control."]
    },
    "Widget": {
        "def": "A widget is a standard component of a graphical user interface (GUI).",
        "part_b": ["Examples include buttons, labels, and text boxes.", "It allows users to interact with the program."],
        "use": "Building UI.",
        "expl": "Think of a widget like a lego brick for app windows. It works by being placed on a window (Input), displaying itself as a button or box (Processing), and waiting for user clicks (Output).",
        "ex": "b = Button(root, text='Click')",
        "app": ["Desktop Apps (Technical).", "Form Submission (Business).", "Calculators (Real-life)."],
        "adv": ["Reusable UI elements.", "Standard look and feel.", "Event driven."]
    },
    "Label": {
        "def": "A label is a widget used to display text or images.",
        "part_b": ["It provides instructions or information to the user.", "It is usually read-only for the user."],
        "use": "Displaying text.",
        "expl": "Think of a label like a sign on a door. It works by taking text (Input), painting it on the window background (Processing), and showing 'Enter Here' to the user (Output).",
        "ex": "l = Label(root, text='Name:')",
        "app": ["Form Headings (Business).", "Status Messages (Technical).", "Image Captions (Real-life)."],
        "adv": ["Simple to use.", "Informative.", "Lightweight."]
    },
    "Event": {
        "def": "An event is an action detected by the program, like a mouse click.",
        "part_b": ["It triggers a specific callback function.", "It drives the logic of GUI applications."],
        "use": "Handling interaction.",
        "expl": "Think of an event like a doorbell ringing. It works by waiting for a user action (Input), triggering a specific 'answering' function (Processing), and executing the code defined for that trigger (Output).",
        "ex": "btn.bind('<Button-1>', on_click)",
        "app": ["Button Clicks (Real-life).", "Key Presses (Technical).", "Mouse Movement (Business)."],
        "adv": ["Responsive apps.", "Interactive.", "Asynchronous."]
    },

    # --- Unit 5: Database (Expanded) ---
    "Cursor": {
        "def": "A cursor is a database object used to traverse and manipulate result sets.",
        "part_b": ["It acts as a pointer to the current row in a query result.", "It allows row-by-row processing."],
        "use": "Navigating results.",
        "expl": "Think of a cursor like your finger on a list. It works by pointing to one item at a time (Input), letting you read or change it (Processing), and then moving down to the next item (Output).",
        "ex": "cur = conn.cursor()",
        "app": ["Processing Query Results (Technical).", "Updating Records (Business).", "Data Migration (Real-life)."],
        "adv": ["Row control.", "Memory efficient.", "Traverses output."]
    },
    "Query": {
        "def": "A query is a request for data or action from a database.",
        "part_b": ["It is usually written in SQL.", "It can select, insert, update, or delete data."],
        "use": "Requesting data.",
        "expl": "Think of a query like an order at a restaurant. It works by taking your specification 'Burger, no onions' (Input), sending it to the kitchen/DB (Processing), and bringing back the exact food you asked for (Output).",
        "ex": "SELECT * FROM users",
        "app": ["Searching Products (Real-life).", "Generating Reports (Business).", "Login Checks (Technical)."],
        "adv": ["Powerful filtering.", "Standardized language.", "Flexible."]
    },
    "SQL": {
        "def": "SQL (Structured Query Language) is the standard language for relational databases.",
        "part_b": ["It is used to define and manipulate data.", "It works across different DB systems like MySQL and SQLite."],
        "use": "Talking to databases.",
        "expl": "Think of SQL like a universal language for filing cabinets. It works by taking English-like commands (Input), converting them into database actions (Processing), and organizing data perfectly (Output).",
        "ex": "CREATE TABLE customers ...",
        "app": ["Data Analysis (Business).", "Backend Web Dev (Technical).", "Banking Systems (Real-life)."],
        "adv": ["Declarative.", "Widely adopted.", "Powerful."]
    },
    "Table": {
        "def": "A table is a collection of related data held in a structured format.",
        "part_b": ["It consists of rows and columns.", "It represents an entity like 'Users' or 'Orders'."],
        "use": "Structuring data.",
        "expl": "Think of a table like a spreadsheet page. It works by defining columns for data types (Input), creating rows for each entry (Processing), and storing everything in a grid (Output).",
        "ex": "CREATE TABLE users (id INT, name TEXT)",
        "app": ["Excel Sheets (Real-life).", "User Directories (Business).", "Log Records (Technical)."],
        "adv": ["Structured.", "Readable.", "Efficient indexing."]
    },
    "Row": {
        "def": "A row represents a single, implicit structured data item in a table.",
        "part_b": ["It is also called a record or tuple.", "It contains data for one specific entity."],
        "use": "Single record storage.",
        "expl": "Think of a row like a single card in a rolodex. It works by holding all the details for one specific person (Input), keeping them grouped together (Processing), and letting you pull that one card out (Output).",
        "ex": "INSERT INTO users VALUES (1, 'John')",
        "app": ["A Contact (Real-life).", "An Order (Business).", "A Student (Academic)."],
        "adv": ["Atomic unit.", "Self-contained.", "Easy update."]
    },

    # --- Reuse ML Terms for multiple units if needed, but ensure distinct sets ---
    "Machine Learning": {
        "def": "Machine Learning is an application of AI that enables systems to learn from data automatically.",
        "part_b": ["It allows computers to solve new problems without specific programming.", "It enables systems to improve their accuracy automatically over time."],
        "use": "Predicting outcomes from data.",
        "expl": "Think of Machine Learning like teaching a child to recognize cats. It works by taking many example photos (Input), analyzing them to find common patterns like ears and whiskers (Processing), and creating a model that can identify cats in new photos (Output). It learns from experience rather than strict rules.",
        "ex": "model.fit(training_data, labels)\nprediction = model.predict(new_data)",
        "app": ["Netflix Movie Recommendations (Real-life).", "Spam Email Filtering (Technical).", "Predicting Stock Trends (Business)."],
        "adv": ["Solves Complex Problems.", "Improves Over Time.", "Adapts to New Data."]
    },
    "Supervised Learning": {
        "def": "Supervised Learning is a type of ML where the model learns from labeled training data.",
        "part_b": ["It requires input data to be paired with the correct output label.", "It is used for classification and regression tasks."],
        "use": "Learning from examples.",
        "expl": "Think of Supervised Learning like a student with an answer key. It works by taking practice questions along with their answers (Input), studying the relationship between them (Processing), and learning to answer similar questions on a test (Output). The 'Supervisor' provides the correct answers during training.",
        "ex": "model.fit(X_train, y_train) # y_train are labels",
        "app": ["Email Spam Detection (Real-life).", "House Price Prediction (Business).", "Handwriting Recognition (Technical)."],
        "adv": ["High Accuracy.", "Clear Performance Metrics.", "Applicable to many problems."]
    },
    "Unsupervised Learning": {
        "def": "Unsupervised Learning involves training on data without labels to find hidden patterns.",
        "part_b": ["It deals with unlabeled data where the outcome is unknown.", "It is often used for clustering and association."],
        "use": "Finding hidden structure.",
        "expl": "Think of Unsupervised Learning like sorting a bucket of miscellaneous legos. It works by taking a pile of mixed bricks (Input), grouping them by similarity like color or size (Processing), and organizing them into clusters (Output). No one told it what 'Red' was; it just noticed they were similar.",
        "ex": "kmeans = KMeans(n_clusters=3)\nkmeans.fit(data)",
        "app": ["Customer Segmentation (Business).", "Genetic Clustering (Academic).", "Anomaly Detection (Technical)."],
        "adv": ["No labeled data needed.", "Finds hidden patterns.", "Useful for exploration."]
    },
    "Neural Network": {
        "def": "A Neural Network is a computational model inspired by the biological neural networks of brains.",
        "part_b": ["It excels at recognizing complex patterns in images and audio.", "It can learn non-linear relationships that simple algorithms miss."],
        "use": "Solving complex pattern problems.",
        "expl": "Think of a Neural Network like a team of detectives. It works by taking complex evidence like an image (Input), passing it through layers of analysis where different features are checked (Processing), and combining these clues to solve the case, like 'This is a Cat' (Output). It mimics the human brain's way of thinking.",
        "ex": "model.add(Dense(10, input_dim=5))",
        "app": ["Siri / Alexa Voice Recognition (Real-life).", "Self-Driving Cars (Technical).", "Language Translation (Academic)."],
        "adv": ["High Accuracy on Images/Voice.", "Handles Messy Data.", "Powerful Pattern Recognition."]
    },
    "Regression": {
        "def": "Regression is a supervised learning technique used to predict continuous numerical values.",
        "part_b": ["It helps in forecasting future trends based on past history.", "It quantifies the relationship between different variables."],
        "use": "Forecasting numerical trends.",
        "expl": "Think of Regression like predicting your travel time. It works by taking past data like distance and traffic (Input), calculating the mathematical relationship or trend line (Processing), and giving you an estimated arrival time (Output). It uses history to look into the future.",
        "ex": "model = LinearRegression()\nmodel.fit(X, y)\nprint(model.predict([[5]]))",
        "app": ["Predicting House Prices (Real-life).", "Estimating Sales Revenue (Business).", "Forecasting Temperature (Academic)."],
        "adv": ["Simple to Understand.", "Fast to Train.", "Gives Clear Numbers."]
    },
    "Classification": {
        "def": "Classification is a supervised learning task to predict a categorical label.",
        "part_b": ["It categorizes input data into distinct classes.", "It produces discrete output (e.g., Yes/No, Red/Blue)."],
        "use": "Categorizing data.",
        "expl": "Think of Classification like a mail sorter. It works by taking a letter (Input), reading the zip code (Processing), and tossing it into the correct city bin (Output). It's all about making a choice between distinct categories.",
        "ex": "clf = LogisticRegression()\nclf.predict(image)",
        "app": ["Spam vs Non-Spam Filter (Real-life).", "Disease Diagnosis (Yes/No) (Healthcare).", "Image Object Detection (Technical)."],
        "adv": ["Clear Categories.", "Widely used application.", "Probabilistic output."]
    },
    "Clustering": {
        "def": "Clustering is an unsupervised task of grouping similar data points together.",
        "part_b": ["It organizes unlabelled data into groups based on similarity.", "It is useful for exploratory data analysis."],
        "use": "Grouping similar items.",
        "expl": "Think of Clustering like organizing a library with no signs. It works by taking a pile of books (Input), comparing their content to see which are similar (Processing), and shelving related books together (Output). You discover the genres as you go.",
        "ex": "kmeans = KMeans()\nkmeans.fit(cistomer_data)",
        "app": ["Market Segmentation (Business).", "Social Network Analysis (Technical).", "Document Clustering (Academic)."],
        "adv": ["No labels usage.", "Insight discovery.", "Natural grouping."]
    },
    "NLP": {
        "def": "Natural Language Processing (NLP) is the interaction between computers and human language.",
        "part_b": ["It enables computers to understand and generate human text.", "It bridges the gap between machine code and human communication."],
        "use": "Processing human language.",
        "expl": "Think of NLP like a universal translator. It works by taking human speech or text (Input), analyzing the grammar, context, and meaning using algorithms (Processing), and converting it into something the computer can act on or reply to (Output). It allows us to talk to machines.",
        "ex": "tokens = nltk.word_tokenize('Hello World')",
        "app": ["Google Translate (Real-life).", "Chatbots (Business).", "Sentiment Analysis (Technical)."],
        "adv": ["Human-Computer Interaction.", "Automates text tasks.", "Analyzes massive text data."]
    }
}

GENERIC_OPTS = {
    "app": ["Used in industry projects.", "Helps optimize code.", "Common in large systems.", "Useful for data analysis."],
    "adv": ["Improves efficiency.", "Widely supported.", "Flexible usage.", "Standard industry practice."],
    "dis": ["Can be complex to learn.", "Requires careful implementation."]
}

# EXPANDED UNIT MAPPING
UNIT_MAP = {
    "Unit 1: Python Basics": ["Variable", "List", "Tuple", "Dictionary", "Set", "Function", "Module", "String", "Integer", "Loop", "Operator", "Comment"],
    "Unit 2: OOP Using Python": ["Class", "Object", "Inheritance", "Polymorphism", "Encapsulation", "Abstraction", "Method", "Constructor", "Attribute"],
    "Unit 3: Plotting & Algorithms": ["List", "Dictionary", "Function", "Array", "Histogram", "Scatter Plot", "Bar Chart", "Legend"], 
    "Unit 4: Network & GUI": ["Class", "Function", "Socket", "Port", "Widget", "Label", "Event", "Module"], 
    "Unit 5: Database Connectivity": ["Variable", "Dictionary", "Cursor", "Query", "SQL", "Table", "Row", "Tuple"],
    "Unit 1: Introduction to ML": ["Machine Learning", "Supervised Learning", "Unsupervised Learning", "Classification", "Regression"],
    "Unit 2: Supervised Learning": ["Regression", "Classification", "Supervised Learning", "Machine Learning"],
    "Unit 3: Unsupervised Learning": ["Clustering", "Unsupervised Learning", "Machine Learning", "Classification"], # re-use closely related
    "Unit 4: NLP": ["NLP", "Neural Network", "Classification", "Machine Learning"],
    "Unit 5: Computer Vision": ["Neural Network", "Classification", "Machine Learning", "Supervised Learning"]
}

def verify_db_integrity():
    """Ensures every term in UNIT_MAP exists in DB."""
    missing = []
    for unit, terms in UNIT_MAP.items():
        for term in terms:
            if term not in DB:
                missing.append(f"{unit}: {term}")
    
    if missing:
        raise ValueError(f"CRITICAL ERROR: The following terms are missing from the DB: {missing}")
    print("Integrity Check Passed: All terms are defined.")

def get_unique_terms_only(terms, count):
    """
    Returns up to 'count' UNIQUE terms.
    If requested count > available terms, returns all available terms (no repetition).
    """
    if count > len(terms):
        print(f"Warning: Requested {count} items but only {len(terms)} available. Returning {len(terms)} unique items.")
        return random.sample(terms, len(terms)) # Return all shuffled
    return random.sample(terms, count)

def generate_part_a(term):
    t = DB[term]
    # Plain text 1 marks - Already plain text
    return {
        "question": f"Define '{term}'.",
        "answer": t["def"]
    }

def generate_part_b(term):
    t = DB[term]
    points = t.get("part_b", [t["use"], t["app"][0]])
    # Plain text 2 marks - List format
    return {
        "question": f"State two distinct features or uses of '{term}'.",
        "answer": f"1. {points[0]}\n2. {points[1]}"
    }

def generate_part_c(term):
    t = DB[term]
    # Full explanation: What/How/Why (from DB 'expl') + Use case
    explanation = f"{t['expl']} It is primarily used for {t['use'].lower()}"
    # Plain text 3 marks - REMOVED MARKDOWN (**), ADDED NUMBERED SECTIONS, PLAIN CODE
    return {
        "question": f"Explain '{term}' with an example.",
        "answer": f"""1. Definition:
{t['def']}

2. Explanation:
{explanation}

3. Example:
{t['ex']}"""
    }

def generate_part_d(term):
    # 5 Marks - STRICT PLAIN TEXT
    t = DB[term]
    
    apps = t.get("app", []) + GENERIC_OPTS["app"]
    advs = t.get("adv", []) + GENERIC_OPTS["adv"]
    
    # \n for newlines, NO // marks
    app_str = "\n".join([f"{i+1}. {x}" for i, x in enumerate(apps[:3])]) 
    adv_str = "\n".join([f"{i+1}. {x}" for i, x in enumerate(advs[:3])])
    
    return {
        "question": f"Discuss '{term}' in detail. Include its advantages and applications.",
        "answer": f"""1. Definition:
{t['def']}

2. Explanation:
{t['expl']}

3. Example:
{t['ex']}

4. Applications:
{app_str}

5. Advantages:
{adv_str}"""
    }

def generate_data():
    verify_db_integrity()
    
    data = []
    for unit in UNITS:
        # Get terms valid for this unit
        possible_terms = UNIT_MAP.get(unit, [])
        if not possible_terms:
            continue
        
        sections = {
            "Part A (1-Mark)": [],
            "Part B (2-Mark)": [],
            "Part C (3-Mark)": [],
            "Part D (5-Mark)": []
        }
        
        # INCREASED VOLUME: Limit to unique terms available
        # Max is 20, but strict no repetition means we cap at len(possible_terms)
        
        # A: Max 20 unique
        a_terms = get_unique_terms_only(possible_terms, 20)
        for term in a_terms:
            sections["Part A (1-Mark)"].append(generate_part_a(term))
            
        # B: Max 20 unique (fresh shuffle)
        b_terms = get_unique_terms_only(possible_terms, 20)
        for term in b_terms:
             sections["Part B (2-Mark)"].append(generate_part_b(term))
             
        # C: Max 20 unique
        c_terms = get_unique_terms_only(possible_terms, 20)
        for term in c_terms:
             sections["Part C (3-Mark)"].append(generate_part_c(term))
             
        # D: Max 20 unique
        d_terms = get_unique_terms_only(possible_terms, 20)
        for term in d_terms:
             sections["Part D (5-Mark)"].append(generate_part_d(term))

        data.append({
            "unit": unit,
            "sections": sections
        })
    return data

if __name__ == "__main__":
    output_path = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/exam_questions.json'
    try:
        data = generate_data()
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully generated exam questions.")
    except Exception as e:
        print(f"Error: {e}")
        # Crash explicitly if integrity match fails
        sys.exit(1)
