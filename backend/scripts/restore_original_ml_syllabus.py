import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')

# Original Syllabus Structure with ENHANCED Explanations (Rich HTML)
# Unit 1: Introduction to ML
# Unit 2: Supervised Learning
# Unit 3: Unsupervised Learning
# Unit 4: NLP
# Unit 5: Computer Vision with OpenCV

original_ml_tutorials = [
    # --- UNIT 1: INTRODUCTION TO ML ---
    {
        "id": "ml-unit1-intro",
        "title": "1. What is Machine Learning?",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The science of getting computers to act without being explicitly programmed.",
        "description": "Introduction to the core concepts of ML, AI, and Deep Learning hierarchy.",
        "explanation": "<h2>1. Definition</h2><p>Machine Learning (ML) is a subset of Artificial Intelligence (AI) that focuses on building systems that learn from data. Instead of explicitly programming rules (if-then), we feed data to algorithms to learn patterns.</p><h2>2. AI vs ML vs DL</h2><ul><li><strong>AI:</strong> The broad concept of smart machines.</li><li><strong>ML:</strong> Getting computers to learn from data.</li><li><strong>DL:</strong> A subset of ML using Neural Networks.</li></ul><h2>3. The ML Workflow</h2><p>Data Collection -> Preprocessing -> Training models -> Evaluation -> Prediction.</p>",
        "key_points": ["AI vs ML", "Workflow", "Data Driven"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit1-types",
        "title": "2. Types of Machine Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Supervised, Unsupervised, and Reinforcement Learning.",
        "description": "Understanding the three main paradigms of learning.",
        "explanation": "<h2>1. Supervised Learning</h2><p>Learning with a teacher. The data is labeled (Input X + Output Y). Example: Spam Classification.</p><h2>2. Unsupervised Learning</h2><p>Learning without a teacher. Data is unlabeled. The goal is to find hidden structure. Example: Customer Segmentation.</p><h2>3. Reinforcement Learning</h2><p>Learning by trial and error based on rewards/punishment. Example: Robots learning to walk.</p>",
        "key_points": ["Labeled Data", "Unlabeled Data", "Rewards"],
        "try_it_yourself": True
    },
    
    # --- UNIT 2: SUPERVISED LEARNING ---
    {
        "id": "ml-unit2-linear",
        "title": "1. Linear Regression",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Predicting a continuous value.",
        "description": "The simplest regression algorithm fitting a straight line.",
        "explanation": "<h2>1. Concept</h2><p>Linear Regression attempts to model the relationship between two variables by fitting a linear equation (y = mx + c) to observed data.</p><h2>2. Use Cases</h2><p>Predicting house prices, stock trends, or temperature.</p><h2>3. Math</h2><p>The goal is to minimize the sum of squared errors the distance between the points and the line.</p>",
        "key_points": ["Regression", "Best Fit Line", "y=mx+c"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-logistic",
        "title": "2. Logistic Regression",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Predicting a category (Yes/No).",
        "description": "Used for binary classification problems.",
        "explanation": "<h2>1. Concept</h2><p>Despite the name, it is a classification algorithm. It uses the Sigmoid function to squash output between 0 and 1, representing probability.</p><h2>2. Sigmoid Function</h2><p>$$ S(x) = \\frac{1}{1 + e^{-x}} $$</p><h2>3. Threshold</h2><p>If probability > 0.5, class is 1 (Yes); else class is 0 (No).</p>",
        "key_points": ["Classification", "Sigmoid", "Probability"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-knn",
        "title": "3. K-Nearest Neighbors (KNN)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Lazy Learning algorithm.",
        "description": "Classifying based on the majority vote of neighbors.",
        "explanation": "<h2>1. Concept</h2><p>To classify a new point, look at its 'K' closest neighbors in the training data. If most are Red, the new point is Red.</p><h2>2. Choosing K</h2><p>If K is too small, it's sensitive to noise. If K is too large, it misses local patterns. Odd numbers are preferred to avoid ties.</p>",
        "key_points": ["Distance Metric", "Lazy Learner", "Voting"],
        "try_it_yourself": True
    },
    
    # --- UNIT 3: UNSUPERVISED LEARNING ---
    {
        "id": "ml-unit3-kmeans",
        "title": "1. K-Means Clustering",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Partitioning data into K clusters.",
        "description": "Iterative algorithm to group similar data points.",
        "explanation": "<h2>1. Algorithm</h2><p>1. Initialize K centroids randomly.<br>2. Assign each point to the nearest centroid.<br>3. Recalculate centroids.<br>4. Repeat until convergence.</p><h2>2. Elbow Method</h2><p>A technique to find the optimal number of clusters (K) by plotting WCSS (Within-Cluster Sum of Square).</p>",
        "key_points": ["Centroids", "Clustering", "Iteration"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-pca",
        "title": "2. Dimensionality Reduction (PCA)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Reducing features while keeping information.",
        "description": "Principal Component Analysis basics.",
        "explanation": "<h2>1. The Curse of Dimensionality</h2><p>Too many features (columns) can make models slow and prone to overfitting.</p><h2>2. PCA Concept</h2><p>PCA transforms the data into new variables (Principal Components) that explain the most variance (information) with fewer dimensions.</p>",
        "key_points": ["Features", "Variance", "Compression"],
        "try_it_yourself": True
    },

    # --- UNIT 4: NATURAL LANGUAGE PROCESSING (NLP) ---
    {
        "id": "ml-unit4-intro",
        "title": "1. Intro to NLP",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Interactions between computers and human language.",
        "description": "Analyzing and generating text data.",
        "explanation": "<h2>1. What is NLP?</h2><p>It enables machines to understand, interpret, and generate human language. Used in Chatbots, Translation, and Sentiment Analysis.</p><h2>2. Libraries</h2><p><strong>NLTK:</strong> The classic library for teaching and research.<br><strong>SpaCy:</strong> Industrial-strength NLP.</p>",
        "key_points": ["Text Analysis", "Linguistics", "NLTK"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-preprocessing",
        "title": "2. Text Preprocessing",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Cleaning text for machines.",
        "description": "Tokenization, Stopwords, and Stemming.",
        "explanation": "<h2>1. Tokenization</h2><p>Splitting text into words (tokens).</p><h2>2. Stopwords</h2><p>Removing common words like 'the', 'is', 'at' that don't add much meaning.</p><h2>3. Stemming/Lemmatization</h2><p>Reducing words to their root form (e.g., 'Running' -> 'Run').</p>",
        "key_points": ["Cleaning", "Tokens", "Roots"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-bow",
        "title": "3. Bag of Words & TF-IDF",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Converting text to numbers.",
        "description": "Feature extraction techniques for text.",
        "explanation": "<h2>1. Bag of Words (BoW)</h2><p>Creates a matrix where columns are words and rows are document counts. Ignores grammar/order.</p><h2>2. TF-IDF</h2><p>Term Frequency-Inverse Document Frequency. Gives weight to unique words and down-weights common ones (like 'is', 'the').</p>",
        "key_points": ["Vectorization", "Frequency", "Matrix"],
        "try_it_yourself": True
    },

    # --- UNIT 5: COMPUTER VISION ---
    {
        "id": "ml-unit5-intro",
        "title": "1. Intro to Computer Vision",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Teaching computers to 'see'.",
        "description": "Processing and analyzing images.",
        "explanation": "<h2>1. What is Computer Vision?</h2><p>AI field that enables computers to derive meaningful information from digital images/videos.</p><h2>2. Images as Data</h2><p>To a computer, an image is just a grid of numbers (Pixel values from 0-255). 0=Black, 255=White.</p>",
        "key_points": ["Pixels", "RGB", "OpenCV"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-opencv",
        "title": "2. OpenCV Basics",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "The standard CV library.",
        "description": "Reading, displaying, and editing images.",
        "explanation": "<h2>1. Loading Images</h2><p><code>cv2.imread('image.jpg')</code> loads an image into a NumPy array.</p><h2>2. Grayscale</h2><p>Converting to black and white (<code>cv2.COLOR_BGR2GRAY</code>) simplifies the data for detection algorithms.</p>",
        "key_points": ["cv2", "imread", "Grayscale"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-face",
        "title": "3. Face Detection",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Haar Cascade Classifiers.",
        "description": "Detecting faces in real-time.",
        "explanation": "<h2>1. Haar Cascades</h2><p>A machine learning based approach where a cascade function is trained from a lot of positive and negative images.</p><h2>2. Implementation</h2><p>OpenCV provides pre-trained XML files. We just load the XML and call <code>detectMultiScale()</code> to get the coordinates of the face.</p>",
        "key_points": ["Object Detection", "XML Models", "Bounding Box"],
        "try_it_yourself": True
    }
]

def restore_tutorials():
    print("Restoring Original ML Syllabus...")
    with open(tutorials_path, 'r') as f:
        data = json.load(f)
    
    # 1. Keep all Non-ML content (Python, etc.)
    # 2. Filter out the NEW "Deep Learning" content we added (ids starting with dl-)
    # 3. Filter out any remnant ML content to be safe
    
    clean_data = [
        t for t in data 
        if not t.get('id').startswith('dl-') and t.get('technology') != "Machine Learning with Python"
    ]
    
    # Append the Restored Original ML Syllabus
    clean_data.extend(original_ml_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print("Tutorials Restored to 5-Unit ML Syllabus.")

if __name__ == "__main__":
    restore_tutorials()
