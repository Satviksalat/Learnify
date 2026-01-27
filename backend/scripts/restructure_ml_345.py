import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

NEW_UNIT_3 = [
    {
        "id": "ml-unit3-intro",
        "title": "1. Introduction to Unsupervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Learning without Labels",
        "description": "Clustering vs Dimensionality Reduction.",
        "syntax": "Theory",
        "code_example": "# No Labels (y)\n# Only Features (X)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Unlabeled Data", "Pattern Discovery", "Clustering"]
    },
    {
        "id": "ml-unit3-clustering-overview",
        "title": "2. Clustering – Overview",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Grouping Data",
        "description": "Distance measures and applications.",
        "syntax": "Theory",
        "code_example": "# Grouping similar items",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Euclidean Distance", "Manhattan Distance", "Similarity"]
    },
    {
        "id": "ml-unit3-kmeans-theory",
        "title": "3. K-Means Clustering – Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Centroid-based",
        "description": "Iterative grouping.",
        "syntax": "KMeans(n_clusters=k)",
        "code_example": "from sklearn.cluster import KMeans\nmodel = KMeans(n_clusters=3)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Centroids", "Elbow Method", "K value"]
    },
    {
        "id": "ml-unit3-kmeans-case",
        "title": "4. K-Means Clustering – Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Python Implementation",
        "description": "Clustering a dataset.",
        "syntax": "model.fit(X)",
        "code_example": "model.fit(X)\nlabels = model.predict(X)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Loading", "Fitting", "Visualizing"]
    },
    {
        "id": "ml-unit3-img-compression",
        "title": "5. Image Compression via Vector Quantization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Color Reduction",
        "description": "Using clustering to reduce colors.",
        "syntax": "KMeans for Colors",
        "code_example": "# 16 Million Colors -> 16 Colors",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Compression", "Pixel Grouping", "Optimization"]
    },
    {
        "id": "ml-unit3-meanshift",
        "title": "6. Mean Shift Clustering",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Sliding Window",
        "description": "Kernel Density Estimation.",
        "syntax": "MeanShift()",
        "code_example": "from sklearn.cluster import MeanShift\nms = MeanShift()",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["No K needed", "Density Peaks", "Slower"]
    },
    {
        "id": "ml-unit3-agglomerative",
        "title": "7. Agglomerative Clustering",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Hierarchical",
        "description": "Bottom-up approach.",
        "syntax": "AgglomerativeClustering()",
        "code_example": "from sklearn.cluster import AgglomerativeClustering",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Dendrogram", "Linkage", "Hierarchy"]
    },
    {
        "id": "ml-unit3-comparison",
        "title": "8. Clustering – Comparative Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Pros and Cons",
        "description": "K-Means vs Mean Shift vs Agglomerative.",
        "syntax": "Theory",
        "code_example": "# Speed vs Accuracy trade-offs",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Scalability", "Shape of clusters", "Parameter tuning"]
    },
    {
        "id": "ml-unit3-summary",
        "title": "9. Unsupervised Learning – Summary",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Recap",
        "description": "Review and Practice.",
        "syntax": "Review",
        "code_example": "# Summary",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Clustering Types", "Use Cases", "Interview Qs"]
    }
]

NEW_UNIT_4 = [
    {
        "id": "ml-unit4-intro",
        "title": "1. Introduction to NLP",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Text Analysis",
        "description": "What is NLP?",
        "syntax": "Theory",
        "code_example": "# Text -> Meaning",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Language Understanding", "Chatbots", "Translation"]
    },
    {
        "id": "ml-unit4-pre-overview",
        "title": "2. Text Preprocessing – Overview",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Cleaning Text",
        "description": "Noise removal pipeline.",
        "syntax": "Theory",
        "code_example": "# Raw Text -> Clean Tokens",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Noise", "Pipeline", "Standardization"]
    },
    {
        "id": "ml-unit4-cleaning",
        "title": "3. Text Cleaning & Tokenization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Breaking down text",
        "description": "Stopwords and punctuation.",
        "syntax": "nltk.word_tokenize()",
        "code_example": "import nltk\ntokens = nltk.word_tokenize('Hello World')",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Tokenization", "Stopwords", "Lowercasing"]
    },
    {
        "id": "ml-unit4-stemming",
        "title": "4. Stemming",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Root Form (Crude)",
        "description": "Porter Stemmer.",
        "syntax": "PorterStemmer()",
        "code_example": "from nltk.stem import PorterStemmer\nps = PorterStemmer()\nprint(ps.stem('running'))",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Chopping suffixes", "Fast", "Not always real words"]
    },
    {
        "id": "ml-unit4-lemmatization",
        "title": "5. Lemmatization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Root Form (Linguistic)",
        "description": "Dictionary based.",
        "syntax": "WordNetLemmatizer()",
        "code_example": "from nltk.stem import WordNetLemmatizer\nlem = WordNetLemmatizer()",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Context aware", "POS Tags", "Slower than Stemming"]
    },
    {
        "id": "ml-unit4-chunking",
        "title": "6. Chunking",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Phrases",
        "description": "Noun Phrase Chunking.",
        "syntax": "RegexpParser()",
        "code_example": "# Grouping (Adjective + Noun)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["NP Chunking", "Shallow Parsing", "Structure"]
    },
    {
        "id": "ml-unit4-vectorization",
        "title": "7. Text Vectorization",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Text to Numbers",
        "description": "BoW and TF-IDF.",
        "syntax": "TfidfVectorizer()",
        "code_example": "from sklearn.feature_extraction.text import TfidfVectorizer",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Bag of Words", "TF-IDF", "Numeric Rep"]
    },
    {
        "id": "ml-unit4-classifier",
        "title": "8. Building a Text Classifier",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Modeling",
        "description": "Pipeline for classification.",
        "syntax": "Pipeline()",
        "code_example": "# Vectorizer -> Classifier",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Features", "Labels", "Training"]
    },
    {
        "id": "ml-unit4-case",
        "title": "9. NLP Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Spam Detection",
        "description": "Full Python example.",
        "syntax": "Implementation",
        "code_example": "# Load Spam Dataset -> Clean -> Train NaiveBayes",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Real World", "Accuracy", "Interpretation"]
    },
    {
        "id": "ml-unit4-summary",
        "title": "10. NLP – Summary",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Recap",
        "description": "Review and Practice.",
        "syntax": "Review",
        "code_example": "# Summary",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Preprocessing Steps", "Vectorization", "Modeling"]
    }
]

NEW_UNIT_5 = [
    {
        "id": "ml-unit5-intro",
        "title": "1. Introduction to Computer Vision",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Visual Perception",
        "description": "Why OpenCV?",
        "syntax": "Theory",
        "code_example": "# Computers 'seeing'",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Pixels", "Matrices", "Applications"]
    },
    {
        "id": "ml-unit5-opencv-basics",
        "title": "2. Introduction to OpenCV",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Library Basics",
        "description": "Reading images and videos.",
        "syntax": "cv2.imread()",
        "code_example": "import cv2\nimg = cv2.imread('image.jpg')",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Installation", "BGR format", "WaitKey"]
    },
    {
        "id": "ml-unit5-haar-theory",
        "title": "3. Haar Cascades – Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Feature Extraction",
        "description": "Viola-Jones Algorithm.",
        "syntax": "CascadeClassifier()",
        "code_example": "face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Haar Features", "Integral Image", "Cascading"]
    },
    {
        "id": "ml-unit5-face-detect",
        "title": "4. Face Detection",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Finding Faces",
        "description": "Using Haar Cascades.",
        "syntax": "detectMultiScale()",
        "code_example": "faces = face_cascade.detectMultiScale(gray, 1.1, 4)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Scale Factor", "Min Neighbors", "Bounding Box"]
    },
    {
        "id": "ml-unit5-features",
        "title": "5. Eye, Nose, Mouth Detection",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Facial Features",
        "description": "Multi-cascade detection.",
        "syntax": "detectMultiScale()",
        "code_example": "# Detect face -> ROI -> Detect eyes",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["ROI (Region of Interest)", "Nested Detection", "Efficiency"]
    },
    {
        "id": "ml-unit5-obj-img",
        "title": "6. Object Detection in Images",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Static Detection",
        "description": "Tuning parameters.",
        "syntax": "Implementation",
        "code_example": "# Tuning minSize and maxSize",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["False Positives", "Parameter Tuning", "Visualization"]
    },
    {
        "id": "ml-unit5-obj-vid",
        "title": "7. Object Detection in Videos",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Real-time Processing",
        "description": "Webcam capture.",
        "syntax": "VideoCapture(0)",
        "code_example": "cap = cv2.VideoCapture(0)\nwhile True: ret, frame = cap.read()",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Frame Loop", "Latency", "Release"]
    },
    {
        "id": "ml-unit5-tracking",
        "title": "8. Face Tracking in Real-Time",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Live Tracking",
        "description": "Drawing boxes on live video.",
        "syntax": "cv2.rectangle()",
        "code_example": "cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Continuous Detection", "Drawing", "User Interface"]
    },
    {
        "id": "ml-unit5-pupil",
        "title": "9. Pupil Detection (Advanced)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Detailed Feature Extraction",
        "description": "Thresholding and Contours.",
        "syntax": "cv2.findContours()",
        "code_example": "thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Thresholding", "Contours", "Centroids"]
    },
    {
        "id": "ml-unit5-case",
        "title": "10. OpenCV Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Full App",
        "description": "Face & Eye detection app.",
        "syntax": "Application",
        "code_example": "# Putting it all together",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Integration", "Performance", "Usability"]
    },
    {
        "id": "ml-unit5-summary",
        "title": "11. Computer Vision – Summary",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Recap",
        "description": "Review and Practice.",
        "syntax": "Review",
        "code_example": "# Summary",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Haar Cascades", "Real-Time", "Image Processing"]
    }
]

def restructure():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 1. Clean out OLD ML Units 3, 4, 5
    # We keep everything that IS NOT part of these units
    cleaned_data = []
    for t in data:
        u = t.get('unit', '')
        if u in ["Unit 3: Unsupervised Learning", "Unit 4: Natural Language Processing", "Unit 5: Computer Vision with OpenCV"]:
            continue
        # Also handle old unit names if they differ slightly
        if "Unsupervised" in u or "Natural Language" in u or "Computer Vision" in u:
            # Check technology to be sure it's ML
            if t.get('technology') == "Machine Learning with Python":
                continue
        cleaned_data.append(t)

    # 2. Append New Units
    cleaned_data.extend(NEW_UNIT_3)
    cleaned_data.extend(NEW_UNIT_4)
    cleaned_data.extend(NEW_UNIT_5)

    with open(JSON_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Successfully restructured ML Units 3, 4, 5.")

if __name__ == "__main__":
    restructure()
