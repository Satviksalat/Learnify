import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')

# THE EXACT SYLLABUS PROVIDED BY USER
syllabus_structure = {
    "Unit 1: Introduction to ML": [
        "1. What is Machine Learning?",
        "2. AI vs ML vs DL",
        "3. Learning from Data",
        "4. The 7 Steps of ML",
        "5. Types of Learning (Overview)",
        "6. Supervised Learning",
        "7. Unsupervised Learning",
        "8. Reinforcement Learning",
        "9. Real-World Applications",
        "10. Unit Summary"
    ],
    "Unit 2: Supervised Learning": [
        "1. Introduction to Supervised Learning",
        "2. Data Preprocessing Overview",
        "3. Mean Removal (Standardization)",
        "4. Scaling & Normalization",
        "5. Binarization & Label Encoding",
        "6. Linear Regression Basics",
        "7. Linear Regression Case Study",
        "8. Introduction to Classification",
        "9. Building a Simple Classifier",
        "10. Logistic Regression Classifier",
        "11. Naive Bayes Classifier",
        "12. Training & Testing Dataset",
        "13. Accuracy & Cross-Validation",
        "14. Confusion Matrix Visualization",
        "15. Classification Performance Report",
        "16. Predictive Modeling Introduction",
        "17. Support Vector Machine (SVM) Theory",
        "18. SVM Linear & Non-Linear Classifiers",
        "19. Confidence Measurements in SVM",
        "20. SVM Case Study",
        "21. Supervised Learning Summary"
    ],
    "Unit 3: Unsupervised Learning": [
        "1. Introduction to Unsupervised Learning",
        "2. Clustering Overview",
        "3. K-Means Clustering Theory",
        "4. K-Means Clustering Case Study",
        "5. Image Compression via Vector Quantization",
        "6. Mean Shift Clustering",
        "7. Agglomerative Clustering",
        "8. Clustering Comparative Study",
        "9. Unsupervised Learning Summary",
        "10. Semi-Supervised Learning"
    ],
    "Unit 4: Natural Language Processing": [
        "1. Introduction to NLP",
        "2. Text Preprocessing Overview",
        "3. Text Cleaning & Tokenization",
        "4. Stemming",
        "5. Lemmatization",
        "6. Chunking",
        "7. Text Vectorization",
        "8. Building a Text Classifier",
        "9. NLP Case Study",
        "10. NLP Summary"
    ],
    "Unit 5: Computer Vision with OpenCV": [
        "1. Introduction to Computer Vision",
        "2. Introduction to OpenCV",
        "3. Haar Cascades Theory",
        "4. Face Detection",
        "5. Eye, Nose, Mouth Detection",
        "6. Object Detection in Images",
        "7. Object Detection in Videos",
        "8. Face Tracking in Real-Time",
        "9. Pupil Detection (Advanced)",
        "10. OpenCV Case Study",
        "11. Computer Vision Summary"
    ]
}

# HELPER: Content Generator for Detailed Descriptions
def generate_content(unit, title):
    # Default Content Structure
    content = {
        "title": title,
        "technology": "Machine Learning with Python",
        "unit": unit,
        "definition": f"Detailed study of {title}.",
        "description": f"Comprehensive guide to {title} in the context of {unit}.",
        "explanation": "",
        "try_it_yourself": True,
        "key_points": ["Key Concept 1", "Key Concept 2", "Key Concept 3"],
        "code_example": "# Example Code\nimport numpy as np\nprint('Hello ML')"
    }
    
    # --- CUSTOM CONTENT MAPPING ---
    
    # UNIT 1
    if "What is Machine Learning" in title:
        content["explanation"] = "<h2>1. Definition</h2><p>Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed (Arthur Samuel, 1959).</p><h2>2. Modern Definition</h2><p>A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E (Tom Mitchell, 1997).</p>"
    elif "AI vs ML vs DL" in title:
        content["explanation"] = "<h2>The Hierarchy</h2><ul><li><strong>AI (Artificial Intelligence):</strong> The broad concept of machines being able to carry out tasks in a way that we would consider 'smart'.</li><li><strong>ML (Machine Learning):</strong> An application of AI based around the idea that we should really just be able to give machines access to data and let them learn for themselves.</li><li><strong>DL (Deep Learning):</strong> A subset of ML inspired by the structure and function of the brain called artificial neural networks.</li></ul>"
    elif "7 Steps of ML" in title:
        content["explanation"] = "<h2>The 7 Steps</h2><ol><li>Data Gathering</li><li>Data Preprocessing</li><li>Choose Model</li><li>Train Model</li><li>Test Model</li><li>Tune Parameters</li><li>Prediction</li></ol>"

    # UNIT 2
    elif "Linear Regression" in title:
         content["code_example"] = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)"
         content["explanation"] = "<h2>1. Introduction</h2><p>Linear Regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables.</p><h2>2. The Equation</h2><p>$$ y = mx + c $$</p><p>Where <strong>m</strong> is the slope and <strong>c</strong> is the intercept.</p>"
    elif "Confusion Matrix" in title:
        content["explanation"] = "<h2>1. What is it?</h2><p>A table that is often used to describe the performance of a classification model.</p><h2>2. Quadrants</h2><ul><li><strong>TP:</strong> True Positive</li><li><strong>TN:</strong> True Negative</li><li><strong>FP:</strong> False Positive (Type 1 Error)</li><li><strong>FN:</strong> False Negative (Type 2 Error)</li></ul>"
    elif "Support Vector Machine" in title:
         content["explanation"] = "<h2>1. Concept</h2><p>SVM is a supervised machine learning algorithm which can be used for both classification and regression challenges.</p><h2>2. Hyperplane</h2><p>The goal is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes.</p>"
    
    # UNIT 3
    elif "K-Means" in title:
        content["explanation"] = "<h2>K-Means Algorithm</h2><p>1. Specify number of clusters K.<br>2. Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.<br>3. Keep iterating until there is no change to the centroids.</p>"
        content["code_example"] = "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3)\nkmeans.fit(X)"
    elif "Image Compression" in title:
        content["explanation"] = "<h2>Vector Quantization</h2><p>Using K-Means to reduce the number of colors in an image. If we choose K=64, we reduce a 16-million color image to just 64 colors, significantly saving space.</p>"
    
    # UNIT 4
    elif "Tokenization" in title:
        content["explanation"] = "<h2>1. Tokenization</h2><p>The process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens.</p><h2>2. NLTK Example</h2><p>Input: 'Hello World'<br>Output: ['Hello', 'World']</p>"
        content["code_example"] = "import nltk\nnltk.download('punkt')\nfrom nltk.tokenize import word_tokenize\ntext = 'Hello World.'\nprint(word_tokenize(text))"
    elif "Stemming" in title:
         content["explanation"] = "<h2>Stemming</h2><p>A process where we remove the suffixes from the words. The result might not be a real word.</p><p>Example: 'Running' -> 'Run', 'Caring' -> 'Car'.</p>"
    
    # UNIT 5
    elif "OpenCV" in title:
         content["explanation"] = "<h2>OpenCV (Open Source Computer Vision Library)</h2><p>An open source computer vision and machine learning software library. It was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products.</p>"
         content["code_example"] = "import cv2\nimg = cv2.imread('image.jpg')\ncv2.imshow('Window', img)\ncv2.waitKey(0)"
    elif "Haar Cascades" in title:
         content["explanation"] = "<h2>Viola-Jones Algorithm</h2><p>Haar-like features are digital image features used in object recognition. They owe their name to their intuitive similarity with Haar wavelets.</p>"
    elif "Pupil Detection" in title:
         content["explanation"] = "<h2>Advanced Eye Tracking</h2><p>Detecting the pupil allows for gaze tracking. This involves converting the eye region to thresholded black and white, then finding the 'blob' (centroid) of the black pupil.</p>"
    
    # GENERIC FALLBACK FOR OTHERS (Ensures they are not empty)
    else:
        content["explanation"] = f"<h2>Detailed Overview of {title}</h2><p>This tutorial covers the essential concepts of {title}. We will explore its theoretical foundation and practical implementation using Python.</p><h2>Key Concepts</h2><ul><li>Concept A: Understanding the basics.</li><li>Concept B: Advanced application.</li><li>Concept C: Common pitfalls.</li></ul>"
    
    return content

def create_tutorials():
    new_tutorials = []
    
    # 1. READ EXISTING to keep Python/Other content
    existing_data = []
    if os.path.exists(tutorials_path):
        with open(tutorials_path, 'r') as f:
            existing_data = json.load(f)
    
    # Filter out OLD ML content (anything with 'Machine Learning with Python')
    clean_data = [t for t in existing_data if t.get('technology') != "Machine Learning with Python"]
    
    # 2. GENERATE NEW ML CONTENT
    for unit_name, titles in syllabus_structure.items():
        for i, title in enumerate(titles):
            # Create a unique-ish ID
            clean_title = title.split('. ')[1].lower().replace(' ', '-').replace('&', 'and').replace('(', '').replace(')', '')
            tid = f"ml-{unit_name.split(':')[0].lower().replace(' ', '')}-{clean_title}"
            
            # Generate Data
            t_data = generate_content(unit_name, title)
            
            # Build Object
            tutorial_obj = {
                "id": tid,
                "title": title,
                "technology": "Machine Learning with Python",
                "unit": unit_name,
                "definition": t_data["definition"],
                "description": t_data["description"],
                "syntax": "Theory & Practice",
                "code_example": t_data["code_example"],
                "explanation": t_data["explanation"],
                "try_it_yourself": True,
                "key_points": t_data["key_points"]
            }
            new_tutorials.append(tutorial_obj)

    # 3. MERGE & WRITE
    clean_data.extend(new_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    
    print(f"Successfully created {len(new_tutorials)} tutorials from the exact syllabus.")

if __name__ == "__main__":
    create_tutorials()
