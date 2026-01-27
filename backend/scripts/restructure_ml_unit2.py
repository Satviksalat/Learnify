import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

NEW_STRUCTURE = [
    {
        "id": "ml-unit2-intro",
        "title": "1. Introduction to Supervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Learning with Teacher",
        "description": "Labeled data, Regression vs Classification.",
        "syntax": "Theory",
        "code_example": "# Input + Label -> Model",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Labeled Data", "Regression", "Classification"]
    },
    {
        "id": "ml-unit2-pre-overview",
        "title": "2. Data Preprocessing – Overview",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Cleaning Data",
        "description": "Why preprocessing is important.",
        "syntax": "Theory",
        "code_example": "# Garbage In = Garbage Out",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Quality Issues", "Formatting", "Cleaning"]
    },
    {
        "id": "ml-unit2-mean-removal",
        "title": "3. Mean Removal (Standardization)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "StandardScaler",
        "description": "Zero mean and unit variance.",
        "syntax": "scaler.fit_transform(X)",
        "code_example": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Zero Mean", "Unit Variance", "Standard Scaler"]
    },
    {
        "id": "ml-unit2-scaling",
        "title": "4. Scaling & Normalization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "MinMax Scaling",
        "description": "Scaling to a range [0, 1].",
        "syntax": "scaler.fit_transform(X)",
        "code_example": "from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\nX_norm = scaler.fit_transform(X)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["MinMaxScaler", "Normalization", "Range 0-1"]
    },
    {
        "id": "ml-unit2-binarization",
        "title": "5. Binarization & Label Encoding",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Categorical to Numerical",
        "description": "Encoding text labels.",
        "syntax": "LabelEncoder()",
        "code_example": "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ny_enc = le.fit_transform(['cat', 'dog'])",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["LabelEncoder", "OneHotEncoder", "Binarizer"]
    },
    {
        "id": "ml-unit2-linreg-basics",
        "title": "6. Linear Regression – Basics",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Fitting a Line",
        "description": "y = mx + b theory.",
        "syntax": "Theory",
        "code_example": "y = mx + c",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Equation of Line", "Coefficients", "Intercept"]
    },
    {
        "id": "ml-unit2-linreg-case",
        "title": "7. Linear Regression – Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Practical Implementation",
        "description": "Training and Predicting.",
        "syntax": "model.predict(X)",
        "code_example": "reg = LinearRegression()\nreg.fit(X_train, y_train)\npred = reg.predict(X_test)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Train", "Test", "Predict"]
    },
    {
        "id": "ml-unit2-cls-intro",
        "title": "8. Introduction to Classification",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Discrete Predictions",
        "description": "Binary vs Multi-class.",
        "syntax": "Theory",
        "code_example": "# Spam vs Not Spam",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Binary", "Multi-Class", "Labels"]
    },
    {
        "id": "ml-unit2-simple-cls",
        "title": "9. Building a Simple Classifier",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Workflow",
        "description": "Steps to build a classifier.",
        "syntax": "Workflow",
        "code_example": "# Feature Selection -> Split -> Train",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Features", "Labels", "Training"]
    },
    {
        "id": "ml-unit2-logistic",
        "title": "10. Logistic Regression Classifier",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Sigmoid Function",
        "description": "Binary classification probability.",
        "syntax": "LogisticRegression()",
        "code_example": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X, y)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Sigmoid", "Probability", "Binary"]
    },
    {
        "id": "ml-unit2-naivebayes",
        "title": "11. Naïve Bayes Classifier",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Probabilistic Model",
        "description": "Based on Bayes Theorem.",
        "syntax": "GaussianNB()",
        "code_example": "from sklearn.naive_bayes import GaussianNB\nnb = GaussianNB()",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Bayes Theorem", "Independence", "Prior/Posterior"]
    },
    {
        "id": "ml-unit2-split",
        "title": "12. Training & Testing Dataset",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Data Splitting",
        "description": "Avoiding Overfitting.",
        "syntax": "train_test_split()",
        "code_example": "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Overfitting", "Underfitting", "Generalization"]
    },
    {
        "id": "ml-unit2-accuracy",
        "title": "13. Accuracy & Cross-Validation",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Evaluation Metrics",
        "description": "Beyond simple accuracy.",
        "syntax": "cross_val_score()",
        "code_example": "from sklearn.model_selection import cross_val_score\nscores = cross_val_score(model, X, y, cv=5)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["K-Fold", "CV", "Robustness"]
    },
    {
        "id": "ml-unit2-conf-matrix",
        "title": "14. Confusion Matrix Visualization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Error Analysis",
        "description": "TP, FP, TN, FN.",
        "syntax": "confusion_matrix()",
        "code_example": "from sklearn.metrics import confusion_matrix\nprint(confusion_matrix(y_true, y_pred))",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["True Positive", "False Positive", "Heatmap"]
    },
    {
        "id": "ml-unit2-cls-report",
        "title": "15. Classification Performance Report",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Precision/Recall",
        "description": "F1-Score and Support.",
        "syntax": "classification_report()",
        "code_example": "from sklearn.metrics import classification_report\nprint(classification_report(y_true, y_pred))",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Precision", "Recall", "F1-Score"]
    },
    {
        "id": "ml-unit2-predictive-intro",
        "title": "16. Predictive Modeling – Introduction",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Advanced Modeling",
        "description": "Linear vs Non-linear.",
        "syntax": "Theory",
        "code_example": "# Complex boundaries",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Non-Linearity", "Complexity", "SVM Intro"]
    },
    {
        "id": "ml-unit2-svm-theory",
        "title": "17. Support Vector Machine (SVM) – Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Max Margin",
        "description": "Hyperplanes and Support Vectors.",
        "syntax": "Theory",
        "code_example": "# Finding the widest road",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Hyperplane", "Margin", "Support Vectors"]
    },
    {
        "id": "ml-unit2-svm-types",
        "title": "18. SVM – Linear & Non-Linear Classifiers",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Kernels",
        "description": "Handling non-linear data.",
        "syntax": "SVC(kernel='rbf')",
        "code_example": "from sklearn.svm import SVC\nmodel = SVC(kernel='rbf')",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Linear Kernel", "RBF", "Polynomial"]
    },
    {
        "id": "ml-unit2-svm-confidence",
        "title": "19. Confidence Measurements in SVM",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Decision Function",
        "description": "Distance from hyperplane.",
        "syntax": "decision_function()",
        "code_example": "dist = model.decision_function(X_test)\nprint(dist)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Distance", "Probability", "Confidence"]
    },
    {
        "id": "ml-unit2-svm-case",
        "title": "20. SVM Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "SVM in Action",
        "description": "Full workflow with SVM.",
        "syntax": "Implementation",
        "code_example": "model = SVC().fit(X_train, y_train)\nprint(model.score(X_test, y_test))",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Preprocessing", "Training", "Acccuracy"]
    },
    {
        "id": "ml-unit2-summary",
        "title": "21. Supervised Learning – Summary",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Unit Recap",
        "description": "Review and Practice.",
        "syntax": "Review",
        "code_example": "# Practice makes perfect",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Recap", "Interview Qs", "Projects"]
    }
]

def restructure():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Remove ALL existing ML Unit 2 items
    cleaned_data = [t for t in data if t.get('unit') != "Unit 2: Supervised Learning"]

    # Insert new structure after ML Unit 1
    # Find index of last ML Unit 1 item
    insert_index = len(cleaned_data)
    for i in range(len(cleaned_data) - 1, -1, -1):
        if cleaned_data[i].get('unit') == "Unit 1: Introduction to ML":
            insert_index = i + 1
            break
    
    # Insert new structure
    for item in reversed(NEW_STRUCTURE):
        cleaned_data.insert(insert_index, item)

    with open(JSON_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Successfully restructured ML Unit 2.")

if __name__ == "__main__":
    restructure()
