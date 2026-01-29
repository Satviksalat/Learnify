import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')

# Full ML Syllabus with Detailed Explanations
# Incorporating 'Deep Learning' context where appropriate without changing Unit Titles.

full_ml_tutorials = [
    # --- UNIT 1: INTRODUCTION TO ML ---
    {
        "id": "ml-unit1-intro",
        "title": "1. What is Machine Learning?",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The science of getting computers to act without being explicitly programmed.",
        "description": "Core concepts of AI, ML, and Deep Learning.",
        "explanation": "<h2>1. Definition</h2><p>Machine Learning (ML) is a subset of AI that focuses on building systems that learn from data. Rather than hard-coding rules, we train models.</p><h2>2. The AI -> ML -> DL Hierarchy</h2><p>It's important to understand the relationship:</p><ul><li><strong>Artificial Intelligence (AI):</strong> The broad umbrella of smart machines.</li><li><strong>Machine Learning (ML):</strong> The subset of AI where machines learn from data.</li><li><strong>Deep Learning (DL):</strong> The subset of ML based on Neural Networks (inspired by the human brain), capable of learning from vast amounts of unstructured data.</li></ul>",
        "key_points": ["AI vs ML vs DL", "Data Driven", "Automation"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit1-types",
        "title": "2. Types of Machine Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Supervised, Unsupervised, Semi-Supervised, Reinforcement.",
        "description": "The paradigms of learning.",
        "explanation": "<h2>1. Supervised Learning</h2><p>Learning with labeled data (Input + Output). Used for Prediction and Classification.</p><h2>2. Unsupervised Learning</h2><p>Learning with unlabeled data. Used for Clustering and Pattern Discovery.</p><h2>3. Semi-Supervised</h2><p>A mix of both (small amount of labeled, large amount of unlabeled).</p><h2>4. Reinforcement Learning</h2><p>Learning via rewards and punishments (e.g., training a game bot).</p>",
        "key_points": ["Supervised", "Unsupervised", "Reinforcement"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit1-workflow",
        "title": "3. The ML Workflow",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Steps to build a model.",
        "description": "From Data Collection to Prediction.",
        "explanation": "<h2>1. Data Collection</h2><p>Gathering raw data.</p><h2>2. Preprocessing</h2><p>Cleaning, handling missing values, and normalization.</p><h2>3. Training</h2><p>Fitting the model to the data.</p><h2>4. Evaluation</h2><p>Testing accuracy using metrics.</p><h2>5. Deployment</h2><p>Using the model in the real world.</p>",
        "key_points": ["Preprocessing", "Training", "Evaluation"],
        "try_it_yourself": True
    },

    # --- UNIT 2: SUPERVISED LEARNING ---
    {
        "id": "ml-unit2-linear",
        "title": "1. Linear Regression",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Predicting continuous values.",
        "description": "The best fit line.",
        "explanation": "<h2>1. Concept</h2><p>Finds the linear relationship between X and Y: $$ y = mx + c $$</p><h2>2. Cost Function</h2><p>Minimizes the Mean Squared Error (MSE) between actual and predicted points.</p><h2>3. Application</h2><p>Predicting sales, temperature, or house prices.</p>",
        "key_points": ["Regression", "MSE", "Line of Best Fit"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-logistic",
        "title": "2. Logistic Regression",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Binary Classification.",
        "description": "Predicting probabilities.",
        "explanation": "<h2>1. Concept</h2><p>Used for classification (Yes/No). Uses the Sigmoid function to squash values between 0 and 1.</p><h2>2. Decision Boundary</h2><p>A threshold (usually 0.5) distinguishes the two classes.</p>",
        "key_points": ["Classification", "Sigmoid", "Binary"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-knn",
        "title": "3. K-Nearest Neighbors (KNN)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Instance-based learning.",
        "description": "Classifying by proximity.",
        "explanation": "<h2>1. Logic</h2><p>'Tell me who your friends are, and I'll tell you who you are.' Classifies a point based on the majority class of its K nearest neighbors.</p><h2>2. Distance Metrics</h2><p>Euclidean Distance is most common.</p>",
        "key_points": ["Lazy Learner", "Euclidean", "K-Value"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-svm",
        "title": "4. Support Vector Machines (SVM)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Finding the Hyperplane.",
        "description": "Maximizing the margin.",
        "explanation": "<h2>1. Hyperplane</h2><p>A boundary that best separates likelihoods.</p><h2>2. Margin</h2><p>The distance between the hyperplane and the nearest data points (Support Vectors). SVM maximizes this margin for better generalization.</p>",
        "key_points": ["Hyperplane", "Margin", "Support Vectors"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-trees",
        "title": "5. Decision Trees & Random Forest",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Tree-based models.",
        "description": "Splitting data based on rules.",
        "explanation": "<h2>1. Decision Tree</h2><p>A flowchart-like structure where internal nodes represents tests (e.g., Age > 20) and leafs represent outcomes.</p><h2>2. Random Forest</h2><p>An ensemble of many decision trees (Bagging). More robust and accurate than a single tree.</p>",
        "key_points": ["Entropy", "Nodes", "Ensemble"],
        "try_it_yourself": True
    },

    # --- UNIT 3: UNSUPERVISED LEARNING ---
    {
        "id": "ml-unit3-intro",
        "title": "1. Intro to Unsupervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Finding hidden patterns.",
        "description": "No labels provided.",
        "explanation": "<h2>1. Goal</h2><p>To discover the underlying structure of the data.</p><h2>2. Types</h2><p><strong>Clustering:</strong> Grouping similar items.<br><strong>Association:</strong> Finding rules (e.g., Market Basket Analysis).</p>",
        "key_points": ["No Teacher", "Structure", "Grouping"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-kmeans",
        "title": "2. K-Means Clustering",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Centroid-based clustering.",
        "description": "Partitioning data into K groups.",
        "explanation": "<h2>1. Algorithm</h2><p>assigns points to the nearest centroid and updates the centroid to the mean of the assigned points.</p><h2>2. Choosing K</h2><p>Use the Elbow Method to find the optimal number of clusters.</p>",
        "key_points": ["Centroids", "Elbow Method", "Iterative"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-pca",
        "title": "3. Dimensionality Reduction (PCA)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Simplifying data.",
        "description": "Reducing features while keeping variance.",
        "explanation": "<h2>1. Why?</h2><p>To visualize high-dimensional data and speed up training.</p><h2>2. Principal Components</h2><p>New uncorrelated variables that maximize variance. PC1 captures the most information.</p>",
        "key_points": ["Variance", "Features", "Compression"],
        "try_it_yourself": True
    },

    # --- UNIT 4: NLP ---
    {
        "id": "ml-unit4-intro",
        "title": "1. NLP Fundamentals",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Language understanding.",
        "description": "Interaction between computers and humans.",
        "explanation": "<h2>1. Libraries</h2><p>NLTK, SpaCy.</p><h2>2. Challenges</h2><p>Ambiguity, Sarcasm, and Slang make NLP difficult compared to structured data.</p>",
        "key_points": ["NLTK", "Text", "Ambiguity"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-preprocessing",
        "title": "2. Text Preprocessing",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Cleaning text.",
        "description": "Preparing text for models.",
        "explanation": "<h2>1. Tokenization</h2><p>Breaking text into words.</p><h2>2. Lowecasing</h2><p>Standardizing case.</p><h2>3. Stopwords Removal</h2><p>Removing common words (and, the, is).</p><h2>4. Lemmatization</h2><p>Reducing words to dictionary root (Better -> Good).</p>",
        "key_points": ["Cleaning", "Tokens", "Roots"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-feature",
        "title": "3. Feature Extraction (BoW & TF-IDF)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Text to Numbers.",
        "description": "Vectorization techniques.",
        "explanation": "<h2>1. Bag of Words</h2><p>Simple frequency count.</p><h2>2. TF-IDF</h2><p>Term Frequency-Inverse Document Frequency. Highlights unique/important words by penalizing very common words.</p>",
        "key_points": ["Vectorization", "TF-IDF", "Matrix"],
        "try_it_yourself": True
    },

    # --- UNIT 5: COMPUTER VISION ---
    {
        "id": "ml-unit5-intro",
        "title": "1. Computer Vision Basics",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Analyzing images.",
        "description": "Pixels, Channels, and Colors.",
        "explanation": "<h2>1. Pixel</h2><p>Smallest unit of an image (0-255).</p><h2>2. Color Space</h2><p>RGB vs Grayscale. Grayscale is preferred for detection as it is computationally cheaper.</p>",
        "key_points": ["Pixels", "RGB", "Grayscale"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-opencv",
        "title": "2. OpenCV Operations",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Image Manipulation.",
        "description": "Reading, Resizing, and Edge Detection.",
        "explanation": "<h2>1. Canny Edge Detection</h2><p>Finds edges in an image by looking for sharp changes in intensity.</p><h2>2. Blurring</h2><p>Gaussian Blur smoothens the image to reduce noise.</p>",
        "key_points": ["Edges", "Blur", "Resize"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-face",
        "title": "3. Face Detection (Haar)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Object Detection.",
        "description": "Viola-Jones Algorithm.",
        "explanation": "<h2>1. Haar Cascades</h2><p>A series of simple classifiers (features) applying to image regions.</p><h2>2. Cascade</h2><p>If a region fails the first stage, it's discarded (making it fast). If it passes, it moves to the next.</p>",
        "key_points": ["Cascade", "Fast", "Real-time"],
        "try_it_yourself": True
    }
]

def update_tutorials():
    print("Writing Comprehensive ML Syllabus to tutorials.json...")
    with open(tutorials_path, 'r') as f:
        data = json.load(f)
    
    # Keep Python content, remove ALL ML content (both OLD and NEW DL stuff) to ensure a clean slate for ML
    clean_data = [
        t for t in data 
        if t.get('technology') != "Machine Learning with Python"
    ]
    
    # Add the NEW 'Full' ML Syllabus
    clean_data.extend(full_ml_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print("Success. Added", len(full_ml_tutorials), "ML tutorials across 5 Units.")

if __name__ == "__main__":
    update_tutorials()
