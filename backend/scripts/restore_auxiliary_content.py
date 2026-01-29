import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

programs_path = os.path.join(DATA_DIR, 'programs.json')
quizzes_path = os.path.join(DATA_DIR, 'quizzes.json')
exam_path = os.path.join(DATA_DIR, 'exam_questions.json')

# --- 1. PROGRAMS (Exercises) ---
ml_programs = [
    # UNIT 1
    {
        "id": "ml-u1-p1",
        "course": "Machine Learning",
        "unit": "Unit 1: Introduction to ML",
        "question": "Python Basics for ML: Import numpy and create a 3x3 array of zeros.",
        "code": "import numpy as np\n\n# Create a 3x3 array of zeros\narr = \nprint(arr)",
        "solution": "import numpy as np\narr = np.zeros((3,3))\nprint(arr)"
    },
    {
        "id": "ml-u1-p2",
        "course": "Machine Learning",
        "unit": "Unit 1: Introduction to ML",
        "question": "Data Loading: Use pandas to load a CSV file named 'data.csv'. (Assume file exists).",
        "code": "import pandas as pd\n\n# Load CSV\ndf = \nprint(df.head())",
        "solution": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())"
    },

    # UNIT 2 (Supervised)
    {
        "id": "ml-u2-p1",
        "course": "Machine Learning",
        "unit": "Unit 2: Supervised Learning",
        "question": "Linear Regression: Implement a simple linear regression using sklearn.",
        "code": "from sklearn.linear_model import LinearRegression\nimport numpy as np\n\nX = np.array([[1], [2], [3], [4]])\ny = np.array([2, 4, 6, 8])\n\n# Initialize and Train Model\nmodel = \n\n# Predict for X=5\nprint(model.predict([[5]]))",
        "solution": "model = LinearRegression()\nmodel.fit(X, y)\nprint(model.predict([[5]]))"
    },
    {
        "id": "ml-u2-p2",
        "course": "Machine Learning",
        "unit": "Unit 2: Supervised Learning",
        "question": "SVM Classifier: Create an SVM classifier with a linear kernel.",
        "code": "from sklearn import svm\nX = [[0, 0], [1, 1]]\ny = [0, 1]\n\n# Create Classifier\nclf = \nclf.fit(X, y)\nprint(clf.predict([[2, 2]]))",
        "solution": "clf = svm.SVC(kernel='linear')\nclf.fit(X, y)\nprint(clf.predict([[2, 2]]))"
    },

    # UNIT 3 (Unsupervised)
    {
        "id": "ml-u3-p1",
        "course": "Machine Learning",
        "unit": "Unit 3: Unsupervised Learning",
        "question": "K-Means: Cluster the given data into 2 groups.",
        "code": "from sklearn.cluster import KMeans\nimport numpy as np\nX = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])\n\n# Fit K-Means\nkmeans = \nkmeans.fit(X)\nprint(kmeans.labels_)",
        "solution": "kmeans = KMeans(n_clusters=2)\nkmeans.fit(X)\nprint(kmeans.labels_)"
    },

    # UNIT 4 (NLP)
    {
        "id": "ml-u4-p1",
        "course": "Machine Learning",
        "unit": "Unit 4: Natural Language Processing",
        "question": "Tokenization: Split the sentence into words using NLTK.",
        "code": "import nltk\nnltk.download('punkt')\nfrom nltk.tokenize import word_tokenize\n\ntext = 'Hello, Machine Learning is fun.'\ntokens = \nprint(tokens)",
        "solution": "tokens = word_tokenize(text)\nprint(tokens)"
    },
    {
        "id": "ml-u4-p2",
        "course": "Machine Learning",
        "unit": "Unit 4: Natural Language Processing",
        "question": "Stemming: Reduce 'Running' and 'Jogging' to their root.",
        "code": "from nltk.stem import PorterStemmer\nps = PorterStemmer()\nwords = ['Running', 'Jogging']\n\nstemmed = [ps.stem(w) for w in words]\nprint(stemmed)",
        "solution": "stemmed = [ps.stem(w) for w in words]"
    },

    # UNIT 5 (CV)
    {
        "id": "ml-u5-p1",
        "course": "Machine Learning",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "question": "Read Image: Read 'image.jpg' in Grayscale mode.",
        "code": "import cv2\n\n# Read image as Grayscale (0)\nimg = \nprint(img.shape)",
        "solution": "img = cv2.imread('image.jpg', 0)\nprint(img.shape)"
    }
]

# --- 2. QUIZZES ---
ml_quizzes = [
    # Unit 1
    {"id": "ml-q1-1", "course": "Machine Learning", "unit": "Unit 1: Introduction to ML", "question": "What is the primary goal of Machine Learning?", "options": ["To explicitly program rules", "To learn patterns from data", "To store data", "To draw graphics"], "correct_answer": "To learn patterns from data"},
    {"id": "ml-q1-2", "course": "Machine Learning", "unit": "Unit 1: Introduction to ML", "question": "Which is NOT a type of learning?", "options": ["Supervised", "Unsupervised", "Reinforcement", "Automated"], "correct_answer": "Automated"},
    
    # Unit 2
    {"id": "ml-q2-1", "course": "Machine Learning", "unit": "Unit 2: Supervised Learning", "question": "In Supervised Learning, the data is:", "options": ["Labeled", "Unlabeled", "Missing", "None of these"], "correct_answer": "Labeled"},
    {"id": "ml-q2-2", "course": "Machine Learning", "unit": "Unit 2: Supervised Learning", "question": "Which algorithm is used for Classification?", "options": ["Linear Regression", "Logistic Regression", "K-Means", "Apriori"], "correct_answer": "Logistic Regression"},

    # Unit 3
    {"id": "ml-q3-1", "course": "Machine Learning", "unit": "Unit 3: Unsupervised Learning", "question": "What is Clustering?", "options": ["Predicting values", "Grouping similar items", "Classifying labels", "None"], "correct_answer": "Grouping similar items"},
    {"id": "ml-q3-2", "course": "Machine Learning", "unit": "Unit 3: Unsupervised Learning", "question": "K-Means is a ___ algorithm.", "options": ["Supervised", "Unsupervised", "Reinforcement", "Deep"], "correct_answer": "Unsupervised"},

    # Unit 4
    {"id": "ml-q4-1", "course": "Machine Learning", "unit": "Unit 4: Natural Language Processing", "question": "What does NLP stand for?", "options": ["Natural Logic Processing", "Natural Language Processing", "Neural Language Program", "None"], "correct_answer": "Natural Language Processing"},
    {"id": "ml-q4-2", "course": "Machine Learning", "unit": "Unit 4: Natural Language Processing", "question": "Removing suffixes from words is called:", "options": ["Tokenization", "Stemming", "Stopwords", "Parsing"], "correct_answer": "Stemming"},

    # Unit 5
    {"id": "ml-q5-1", "course": "Machine Learning", "unit": "Unit 5: Computer Vision with OpenCV", "question": "What library is commonly used for CV in Python?", "options": ["NumPy", "Pandas", "OpenCV", "Matplotlib"], "correct_answer": "OpenCV"},
    {"id": "ml-q5-2", "course": "Machine Learning", "unit": "Unit 5: Computer Vision with OpenCV", "question": "Haar Cascades are used for:", "options": ["Object Detection", "Image Compression", "Text Analysis", "Audio Processing"], "correct_answer": "Object Detection"}
]

# --- 3. EXAMS (Resources) ---
ml_exams = [
    {
        "unit": "Unit 1: Introduction to ML",
        "sections": {
            "Part A (1-Mark)": [{"question": "Define Machine Learning.", "answer": "ML is the field of study that gives computers the ability to learn without being explicitly programmed."}],
            "Part B (2-Mark)": [{"question": "Differentiate AI and ML.", "answer": "AI is the broad science of mimicking human abilities, ML is a specific subset that trains a machine how to learn."}],
            "Part C (5-Mark)": [{"question": "Explain the 7 Steps of Machine Learning.", "answer": "1. Data Gathering\n2. Data Preprocessing\n3. Choose Model\n4. Train Model\n5. Test Model\n6. Tune Parameters\n7. Prediction"}]
        }
    },
    {
        "unit": "Unit 2: Supervised Learning",
        "sections": {
            "Part A (1-Mark)": [{"question": "What is Regression?", "answer": "Predicting a continuous output value."}],
            "Part B (2-Mark)": [{"question": "Explain Confusion Matrix.", "answer": "A table used to describe the performance of a classification model (TP, TN, FP, FN)."}],
            "Part C (5-Mark)": [{"question": "Explain SVM and the Kernel Trick.", "answer": "SVM finds the optimal hyperplane. The Kernel Trick projects data into higher dimensions to make it separable."}]
        }
    },
    {
        "unit": "Unit 3: Unsupervised Learning",
        "sections": {
            "Part A (1-Mark)": [{"question": "Define Clustering.", "answer": "Grouping a set of objects in such a way that objects in the same group are more similar to each other."}],
            "Part B (2-Mark)": [{"question": "How does K-Means work?", "answer": "It partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean."}],
            "Part C (5-Mark)": [{"question": "Compare K-Means and Hierarchical Clustering.", "answer": "K-Means is centroid-based and requires K. Hierarchical builds a tree (dendrogram) and does not require K initially."}]
        }
    },
    {
        "unit": "Unit 4: Natural Language Processing",
        "sections": {
            "Part A (1-Mark)": [{"question": "What is a Token?", "answer": "A meaningful unit of text, such as a word or punctuation."}],
            "Part B (2-Mark)": [{"question": "Stemming vs Lemmatization?", "answer": "Stemming chops off ends (fast, crude). Lemmatization finds the dictionary root (slower, accurate)."}],
            "Part C (5-Mark)": [{"question": "Explain the Text Classification Pipeline.", "answer": "Raw Text -> Cleaning -> Tokenization -> Vectorization (TF-IDF) -> Model Training -> Classification."}]
        }
    },
    {
        "unit": "Unit 5: Computer Vision with OpenCV",
        "sections": {
            "Part A (1-Mark)": [{"question": "What is a Pixel?", "answer": "The smallest unit of a digital image."}],
            "Part B (2-Mark)": [{"question": "How does Face Detection work?", "answer": "Using classifiers like Haar Cascades to scan the image for features (eyes, nose, mouth relationships)."}],
            "Part C (5-Mark)": [{"question": "Explain Object Detection vs Tracking.", "answer": "Detection identifies an object in a frame. Tracking locates a moving object over time."}]
        }
    }
]

def restore_aux():
    # 1. Update Programs
    current_progs = []
    if os.path.exists(programs_path):
        with open(programs_path, 'r') as f:
            current_progs = json.load(f)
    
    # Remove old ML/DL content
    current_progs = [p for p in current_progs if p.get('course') != 'Machine Learning']
    current_progs.extend(ml_programs)

    with open(programs_path, 'w') as f:
        json.dump(current_progs, f, indent=4)
    print(f"Restored {len(ml_programs)} Machine Learning Programs.")

    # 2. Update Quizzes
    current_quizzes = []
    if os.path.exists(quizzes_path):
        with open(quizzes_path, 'r') as f:
            current_quizzes = json.load(f)
    
    current_quizzes = [q for q in current_quizzes if q.get('course') != 'Machine Learning']
    current_quizzes.extend(ml_quizzes)

    with open(quizzes_path, 'w') as f:
        json.dump(current_quizzes, f, indent=4)
    print(f"Restored {len(ml_quizzes)} Machine Learning Quizzes.")

    # 3. Update Exams
    current_exams = []
    if os.path.exists(exam_path):
        with open(exam_path, 'r') as f:
            current_exams = json.load(f)
            
    # Remove exams that match our ML Unit names
    new_exams = [e for e in current_exams if "Unit" in e.get('unit', '') and "Introduction to ML" not in e.get('unit', '') and "Supervised" not in e.get('unit', '') and "Unsupervised" not in e.get('unit', '') and "Processing" not in e.get('unit', '') and "OpenCV" not in e.get('unit', '')]
    
    new_exams.extend(ml_exams)

    with open(exam_path, 'w') as f:
        json.dump(new_exams, f, indent=4)
    print(f"Restored {len(ml_exams)} Machine Learning Exam Units.")

if __name__ == "__main__":
    restore_aux()
