const fs = require('fs');
const path = require('path');

// Target file
const OUTPUT_FILE = path.join(__dirname, '../data/tutorials.json');

const languages = ['python', 'ml_python'];

// Helper to create detailed function HTML
const createFunctionSection = (pkg, funcs) => {
    let html = `<h3>${pkg} Function Reference</h3>`;
    funcs.forEach(f => {
        html += `
            <div style="margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px;">
                <h4 style="color: #2c3e50; margin-bottom: 5px;"><code>${f.signature}</code></h4>
                <p><strong>Description:</strong> ${f.desc}</p>
                <p><strong>Characteristics:</strong> ${f.chars}</p>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px;">
                    <pre style="margin:0; color: #333;">${f.example}</pre>
                </div>
            </div>
        `;
    });
    return html;
};

const topics = {
    python: [
        {
            title: "Introduction",
            sub: "Why Python?",
            description: "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            code: "print('Hello, World!')",
            explanation: `
                <h3>What is Python?</h3>
                <p>Python is a popular programming language created by Guido van Rossum, and released in 1991.</p>
                
                <h3>Key Characteristics:</h3>
                <ul>
                    <li><strong>Interpreted:</strong> Executed line-by-line, facilitating debugging.</li>
                    <li><strong>Dynamically Typed:</strong> Variables do not need explicit declaration.</li>
                    <li><strong>Object-Oriented:</strong> Everything in Python is an object.</li>
                </ul>
            `
        },
        // ... (Keeping basic Python relatively concise to focus on ML libraries as requested, but ensuring core concepts are there)
        {
            title: "Data Structures Detail",
            sub: "List, Dict, Set, Tuple",
            description: "A deep dive into Python's core data structures.",
            code: "lst = [1, 2, 3]\ndct = {'a': 1}\nst = {1, 2}\ntpl = (1, 2)",
            explanation: `
                <h3>Comprehensive Comparison</h3>
                <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f2f2f2;">
                        <th>Type</th>
                        <th>Mutability</th>
                        <th>Ordering</th>
                        <th>Duplicates</th>
                        <th>Syntax</th>
                    </tr>
                    <tr>
                        <td><strong>List</strong></td>
                        <td>Mutable (Changeable)</td>
                        <td>Ordered (Indexed)</td>
                        <td>Allowed</td>
                        <td><code>[1, 2]</code></td>
                    </tr>
                    <tr>
                        <td><strong>Tuple</strong></td>
                        <td>Immutable (Fixed)</td>
                        <td>Ordered (Indexed)</td>
                        <td>Allowed</td>
                        <td><code>(1, 2)</code></td>
                    </tr>
                    <tr>
                        <td><strong>Set</strong></td>
                        <td>Mutable</td>
                        <td>Unordered (No Index)</td>
                        <td><strong>Not Allowed</strong></td>
                        <td><code>{1, 2}</code></td>
                    </tr>
                    <tr>
                        <td><strong>Dictionary</strong></td>
                        <td>Mutable</td>
                        <td>Ordered (Py 3.7+)</td>
                        <td>Keys: No, Vals: Yes</td>
                        <td><code>{'k': 'v'}</code></td>
                    </tr>
                </table>
            `
        }
    ],
    ml_python: [
        {
            title: "NumPy Full Guide",
            sub: "Numerical Python",
            description: "NumPy is the fundamental package for scientific computing in Python. It provides a high-performance multidimensional array object.",
            code: "import numpy as np\n\n# Creating Arrays\narr = np.array([[1, 2, 3], [4, 5, 6]])\nprint('Shape:', arr.shape)\nprint('Mean:', np.mean(arr))",
            explanation: `
                <h3>Core Concept: The ndarray</h3>
                <p>The main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers.</p>
                
                ${createFunctionSection('NumPy', [
                {
                    signature: "np.array(object, dtype=None)",
                    desc: "Creates an array from a list or tuple.",
                    chars: "Creates a new copy in memory. Homogeneous data types.",
                    example: "arr = np.array([1, 2, 3], dtype=float)"
                },
                {
                    signature: "np.zeros(shape)",
                    desc: "Returns a new array of given shape and type, filled with zeros.",
                    chars: "Useful for initializing weights/buffers.",
                    example: "z = np.zeros((2, 3)) # 2 rows, 3 cols"
                },
                {
                    signature: "np.arange(start, stop, step)",
                    desc: "Return evenly spaced values within a given interval.",
                    chars: "Similar to Python's range() but returns an array.",
                    example: "a = np.arange(0, 10, 2) # [0, 2, 4, 6, 8]"
                },
                {
                    signature: "np.linspace(start, stop, num)",
                    desc: "Return num evenly spaced samples, calculated over the interval.",
                    chars: "Crucial for plotting graphs (x-axis generation).",
                    example: "x = np.linspace(0, 1, 5) # [0.  0.25 0.5  0.75 1. ]"
                },
                {
                    signature: "arr.reshape(new_shape)",
                    desc: "Gives a new shape to an array without changing its data.",
                    chars: "Must match total number of elements. Returns a view if possible.",
                    example: "arr.reshape(3, 1)"
                },
                {
                    signature: "np.concatenate((a1, a2), axis=0)",
                    desc: "Join a sequence of arrays along an existing axis.",
                    chars: "Axis 0 = Rows (Vertical), Axis 1 = Cols (Horizontal).",
                    example: "np.concatenate((arr1, arr2), axis=0)"
                }
            ])}
            `
        },
        {
            title: "Pandas Full Guide",
            sub: "Data Manipulation",
            description: "Pandas enables data analysis and manipulation through Series and DataFrame structures.",
            code: "import pandas as pd\n\ndf = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})\nprint(df.describe())",
            explanation: `
                <h3>Core Structures</h3>
                <ul>
                    <li><strong>Series:</strong> One-dimensional labeled array (like a column).</li>
                    <li><strong>DataFrame:</strong> Two-dimensional labeled data structure (like a table).</li>
                </ul>

                ${createFunctionSection('Pandas', [
                {
                    signature: "pd.read_csv(filepath, sep=',')",
                    desc: "Read a comma-separated values (csv) file into DataFrame.",
                    chars: "Highly optimized. Supports chunks, diverse delimiters.",
                    example: "df = pd.read_csv('data.csv')"
                },
                {
                    signature: "df.head(n=5)",
                    desc: "Return the first n rows.",
                    chars: "Quick inspection of data structure.",
                    example: "df.head(10)"
                },
                {
                    signature: "df.loc[label]",
                    desc: "Access a group of rows and columns by label(s) or a boolean array.",
                    chars: "Label-based indexing. Inclusive of end bounds.",
                    example: "df.loc[0:5, ['Name', 'Age']]"
                },
                {
                    signature: "df.iloc[index]",
                    desc: "Purely integer-location based indexing for selection by position.",
                    chars: "0-based indexing. Exclusive of end bounds (Pythonic).",
                    example: "df.iloc[0:5, 0:2]"
                },
                {
                    signature: "df.groupby(by)",
                    desc: "Group DataFrame using a mapper or by a Series of columns.",
                    chars: "Used for split-apply-combine operations (Aggregation).",
                    example: "df.groupby('Category').mean()"
                },
                {
                    signature: "df.fillna(value)",
                    desc: "Fill NA/NaN values using the specified method.",
                    chars: "Essential for data cleaning. Can do forward/backward fill.",
                    example: "df.fillna(0, inplace=True)"
                }
            ])}
            `
        },
        {
            title: "Matplotlib Guide",
            sub: "Data Visualization",
            description: "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.",
            code: "import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.show()",
            explanation: `
                <h3>Anatomy of a Plot</h3>
                <p>Everything in Matplotlib is organized in a hierarchy: Figure -> Axes (Plot) -> Elements (Lines, Text).</p>

                ${createFunctionSection('Pyplot (plt)', [
                {
                    signature: "plt.figure(figsize=(w, h))",
                    desc: "Create a new figure.",
                    chars: "Top level container. Size in inches.",
                    example: "plt.figure(figsize=(10, 6))"
                },
                {
                    signature: "plt.plot(x, y, format)",
                    desc: "Plot y versus x as lines and/or markers.",
                    chars: "Versatile. Supports colors ('r'), styles ('--').",
                    example: "plt.plot(x, y, 'r--', label='Sine')"
                },
                {
                    signature: "plt.scatter(x, y)",
                    desc: "A scatter plot of y vs x.",
                    chars: "Great for showing distribution or correlation.",
                    example: "plt.scatter(data['age'], data['salary'])"
                },
                {
                    signature: "plt.hist(x, bins)",
                    desc: "Compute and draw the histogram of x.",
                    chars: "Shows frequency distribution.",
                    example: "plt.hist(data, bins=20)"
                },
                {
                    signature: "plt.xlabel(label) / plt.ylabel()",
                    desc: "Set the label for the x/y-axis.",
                    chars: "Essential for readable graphs.",
                    example: "plt.xlabel('Time (s)')"
                }
            ])}
            `
        },
        {
            title: "Scikit-Learn Guide",
            sub: "Machine Learning API",
            description: "Simple and efficient tools for predictive data analysis.",
            code: "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y)",
            explanation: `
                <h3>Three Step Pattern</h3>
                <ol>
                    <li><strong>Instantiate:</strong> <code>model = Model()</code></li>
                    <li><strong>Fit:</strong> <code>model.fit(X_train, y_train)</code></li>
                    <li><strong>Predict:</strong> <code>model.predict(X_test)</code></li>
                </ol>

                ${createFunctionSection('Sklearn', [
                {
                    signature: "train_test_split(*arrays, test_size)",
                    desc: "Split arrays or matrices into random train and test subsets.",
                    chars: "Prevents overfitting by holding out data.",
                    example: "train_test_split(X, y, test_size=0.2)"
                },
                {
                    signature: "StandardScaler.fit_transform(X)",
                    desc: "Standardize features by removing the mean and scaling to unit variance.",
                    chars: "Crucial for algorithms like KNN, SVM, Neural Nets.",
                    example: "scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)"
                },
                {
                    signature: "LinearRegression()",
                    desc: "Ordinary least squares Linear Regression.",
                    chars: "Baseline model for regression tasks.",
                    example: "model = LinearRegression()"
                },
                {
                    signature: "LogisticRegression()",
                    desc: "Logistic Regression (despite name, used for Classification).",
                    chars: "Outputs probabilities using Sigmoid function.",
                    example: "model = LogisticRegression()"
                },
                {
                    signature: "mean_squared_error(y_true, y_pred)",
                    desc: "Mean squared error regression loss.",
                    chars: "Lower is better. Heavily penalizes outliers.",
                    example: "mse = mean_squared_error(y_test, predictions)"
                },
                {
                    signature: "accuracy_score(y_true, y_pred)",
                    desc: "Accuracy classification score.",
                    chars: "Ratio of correct predictions. Misleading on imbalanced data.",
                    example: "acc = accuracy_score(y_test, predictions)"
                }
            ])}
            `
        }
    ]
};

const generateTutorials = () => {
    let allTutorials = [];

    languages.forEach(lang => {
        const langTopics = topics[lang];
        if (!langTopics) return;

        langTopics.forEach((topic, index) => {
            const id = `${lang}-${topic.title.toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
            let techDisplay = lang === 'ml_python' ? 'Machine Learning' : 'Python';

            const tutorial = {
                id: id,
                title: topic.title,
                technology: lang,
                definition: topic.description,
                description: topic.description,
                syntax: topic.sub,
                code_example: topic.code,
                explanation: topic.explanation,
                try_it_yourself: true,
                key_points: [
                    "Check the Function Reference",
                    "Understand Parameters",
                    "Run the Example Code"
                ]
            };
            allTutorials.push(tutorial);
        });
    });

    console.log(`Generated ${allTutorials.length} tutorials.`);
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allTutorials, null, 4));
    console.log(`Successfully wrote to ${OUTPUT_FILE}`);
};

generateTutorials();
