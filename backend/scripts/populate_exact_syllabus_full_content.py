import json
import os
import re

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')

# Helper to generate rich HTML content for specific topics
def get_tutorial_content(title, unit_name):
    t_lower = title.lower()
    u_lower = unit_name.lower()
    
    # Defaults
    description = f"Learn about {title}."
    explanation = ""
    code_example = "# No code example required for this theory topic."
    key_points = ["Concept", "Application", "Summary"]

    # --- UNIT SUMMARIES (Check these first to catch all summaries) ---
    if "summary" in t_lower or "unit summary" in t_lower:
        if "unit 1" in u_lower:
            description = "Recap of Machine Learning Foundations and Terminology."
            explanation = """
            <h2>Unit 1: Introduction to ML - Recap</h2>
            <p>This unit established the foundational concepts necessary for understanding Machine Learning.</p>
            <h3>Key Terminology</h3>
            <ul>
                <li><strong>AI vs ML vs DL:</strong> The hierarchy of intelligence. AI is the goal, ML is the tool, DL is the specialized technique.</li>
                <li><strong>7 Steps of ML:</strong> Data Gathering, Preprocessing, Model Choice, Training, Testing, Tuning, Prediction.</li>
                <li><strong>Supervised Learning:</strong> Learning with labeled data (Teacher).</li>
                <li><strong>Unsupervised Learning:</strong> Learning with unlabeled data (Self-discovery).</li>
                <li><strong>Reinforcement Learning:</strong> Learning via Interaction (Reward/Penalty).</li>
            </ul>"""
            key_points = ["AI Hierarchy", "ML Pipeline", "Learning Types"]

        elif "unit 2" in u_lower:
            description = "Recap of Supervised Learning: Regression, Classification, and SVM."
            explanation = """
            <h2>Unit 2: Supervised Learning - Recap</h2>
            <p>We dived deep into models that learn from labeled data. We covered both predicting values and predicting categories.</p>
            <h3>Key Terminology</h3>
            <ul>
                <li><strong>Regression:</strong> Predicting continuous values (e.g., Prices). Algorithms: Linear Regression.</li>
                <li><strong>Classification:</strong> Predicting discrete labels (e.g., Spam/Ham). Algorithms: Logistic Regression, Naive Bayes.</li>
                <li><strong>Preprocessing:</strong> Standardization (Mean Removal), Normalization, Label Encoding.</li>
                <li><strong>Evaluation Metrics:</strong> Accuracy, Precision, Recall, F1-Score, Confusion Matrix.</li>
                <li><strong>SVM:</strong> Finding the hyperplane with the maximum margin. key concepts: Support Vectors, Kernel Trick.</li>
            </ul>"""
            key_points = ["Regression", "Classification", "SVM & Kernels"]

        elif "unit 3" in u_lower:
            description = "Recap of Unsupervised Learning: Clustering and Structure Discovery."
            explanation = """
            <h2>Unit 3: Unsupervised Learning - Recap</h2>
            <p>We explored how to find hidden structures in unlabeled data, primarily through Clustering.</p>
            <h3>Key Terminology</h3>
            <ul>
                <li><strong>Clustering:</strong> Grouping similar data points together.</li>
                <li><strong>K-Means:</strong> Centroid-based clustering. Minimizes distance to center. Good for spherical clusters.</li>
                <li><strong>Hierarchical Clustering (Agglomerative):</strong> Building a tree of clusters (Dendrogram). Good for hierarchy.</li>
                <li><strong>Mean Shift:</strong> Density-based clustering using sliding windows.</li>
                <li><strong>Applications:</strong> Customer Segmentation, Image Compression (Vector Quantization).</li>
            </ul>"""
            key_points = ["Clustering", "K-Means", "Hierarchical"]

        elif "unit 4" in u_lower:
            description = "Recap of Natural Language Processing: From Text to Meaning."
            explanation = """
            <h2>Unit 4: NLP - Recap</h2>
            <p>We learned how to make computers understand human language by converting text into numbers (vectors).</p>
            <h3>Key Terminology</h3>
            <ul>
                <li><strong>Tokenization:</strong> Breaking text into words.</li>
                <li><strong>Stemming/Lemmatization:</strong> Reducing words to their root form (Running -> Run).</li>
                <li><strong>Stopwords:</strong> Removing common, low-value words (the, is).</li>
                <li><strong>Vectorization:</strong> Bag of Words (Frequency) and TF-IDF (Importance).</li>
                <li><strong>Sentiment Analysis:</strong> Classifying text emotion (Positive/Negative).</li>
            </ul>"""
            key_points = ["Text Preprocessing", "Vectorization", "Sentiment Analysis"]

        elif "unit 5" in u_lower:
            description = "Recap of Computer Vision: Seeing with OpenCV."
            explanation = """
            <h2>Unit 5: Computer Vision - Recap</h2>
            <p>We enabled machines to 'see' by processing pixel data using OpenCV.</p>
            <h3>Key Terminology</h3>
            <ul>
                <li><strong>OpenCV:</strong> The standard library for image processing.</li>
                <li><strong>Haar Cascades:</strong> Pre-trained classifiers using rectangular features for rapid face detection.</li>
                <li><strong>Object Detection:</strong> Finding WHAT is in an image and WHERE it is (Bounding Box).</li>
                <li><strong>Object Tracking:</strong> Following a detected object across video frames.</li>
                <li><strong>Applications:</strong> Face Detection, Eye Tracking, Drowsiness Detection.</li>
            </ul>"""
            key_points = ["OpenCV", "Haar Cascades", "Detection vs Tracking"]
        
        return { "explanation": explanation, "code_example": code_example, "key_points": key_points, "description": description }

    
    # --- SPECIFIC CHECKS (Order matters!) ---
    
    # Fix for Semi-Supervised (Must be before Supervised)
    if "semi-supervised" in t_lower:
        description = "Understanding the hybrid approach combining labeled and unlabeled data."
        explanation = """<h2>Best of Both Worlds</h2><p>When you have lots of unlabeled data and a small amount of labeled data. You can use the small labeled set to train a model to pseudo-label the rest, then train on the full set.</p>"""

    # --- UNIT 1: INTRO ---
    elif "what is machine learning" in t_lower:
        description = "Definition, core components, and the philosophy of Machine Learning."
        explanation = """<h2>Definition</h2><p>Machine Learning (ML) is the science of getting computers to learn and act like humans do, and improve their learning over time in learning-autonomous fashion, by feeding them data and information in the form of observations and real-world interactions.</p>
        <h2>Key Components</h2><ul><li><strong>Data:</strong> The fuel for ML.</li><li><strong>Features:</strong> Variables in the data.</li><li><strong>Algorithms:</strong> The logic to learn from data.</li></ul>"""
        key_points = ["Definition", "Arthur Samuel", "Tom Mitchell"]
    
    elif "ai vs ml vs dl" in t_lower:
        description = "Distinguishing between Artificial Intelligence, Machine Learning, and Deep Learning."
        explanation = """<h2>The Hierarchy</h2>
        <p>It is crucial to visualize these as concentric circles:</p>
        <ul>
            <li><strong>Artificial Intelligence (AI):</strong> The broadest term. Any technique that enables computers to mimic human intelligence.</li>
            <li><strong>Machine Learning (ML):</strong> A subset of AI that includes techniques that enable machines to improve at tasks with experience.</li>
            <li><strong>Deep Learning (DL):</strong> A subset of ML composed of algorithms that permit software to train itself to perform tasks, like speech and image recognition, by exposing multilayered neural networks to vast amounts of data.</li>
        </ul>"""
        key_points = ["AI", "ML", "DL"]

    elif "learning from data" in t_lower:
        description = "How machines use examples rather than explicit instructions to learn patterns."
        explanation = """<h2>How Machines Learn</h2><p>Traditional programming relies on hard-coded rules. ML relies on <strong>Examples</strong>. By showing a machine 1000 pictures of a cat, it learns to recognize the visual patterns (ears, whiskers) that define a 'cat'.</p>"""
    
    elif "7 steps of ml" in t_lower:
        description = "The standard end-to-end pipeline: from data gathering to prediction."
        explanation = """<h2>The Standard Pipeline</h2>
        <ol>
            <li><strong>Data Gathering:</strong> Collecting raw data.</li>
            <li><strong>Data Preprocessing:</strong> Cleaning and formatting.</li>
            <li><strong>Choose Model:</strong> Selecting the right algorithm.</li>
            <li><strong>Train Model:</strong> Fitting the data.</li>
            <li><strong>Test Model:</strong> Evaluating performance.</li>
            <li><strong>Parameter Tuning:</strong> Optimizing hyperparameters.</li>
            <li><strong>Prediction:</strong> Deploying for real-world use.</li>
        </ol>"""
    
    elif "types of learning" in t_lower:
        description = "Overview of Supervised, Unsupervised, and Reinforcement Learning."
        explanation = """<h2>Three Main Paradigms</h2>
        <ul>
            <li><strong>Supervised:</strong> Learning with labeled data.</li>
            <li><strong>Unsupervised:</strong> Learning with unlabeled data.</li>
            <li><strong>Reinforcement:</strong> Learning via reward/punishment.</li>
        </ul>"""
    
    # UNIT 1 Unsupervised check
    elif "unsupervised learning" in t_lower and "3:" not in t_lower: 
        description = "Learning from unlabeled data to find hidden structures."
        explanation = """<h2>Unsupervised Learning</h2><p>There is no teacher. The model is given a dataset without explicit instructions on what to do with it. It must find patterns and structure on its own.</p><p>Examples: Customer Segmentation, Anomaly Detection.</p>"""

    # UNIT 1 Supervised check
    elif "supervised learning" in t_lower and "2:" not in t_lower: 
        description = "Learning from labeled data: Mapping Inputs to Outputs."
        explanation = """<h2>Supervised Learning</h2><p>Imagine a teacher supervising a student. The teacher provides the questions (Input) and the answers (Labels). The student (Model) learns to map inputs to answers.</p><p>Examples: Spam Filtering, House Price Prediction.</p>"""

    elif "reinforcement learning" in t_lower:
        description = "Learning through trial and error, rewards and punishments."
        explanation = """<h2>Reinforcement Learning (RL)</h2><p>An agent interacts with an environment and learns to maximize a reward. It learns from consequences.</p><p>Example: Training a robot to walk, or an AI to play Chess.</p>"""

    elif "real-world applications" in t_lower:
        description = "Case studies: Healthcare, Finance, Retail, and Social Media."
        explanation = """<h2>ML is Everywhere</h2>
        <ul>
            <li><strong>Healthcare:</strong> Disease diagnosis.</li>
            <li><strong>Finance:</strong> Fraud detection.</li>
            <li><strong>Retail:</strong> Product recommendations.</li>
            <li><strong>Social Media:</strong> Content curation.</li>
            <li><strong>Self-Driving Cars:</strong> Object detection.</li>
        </ul>"""

    # --- UNIT 2: SUPERVISED ---
    elif "introduction to supervised learning" in t_lower:
        description = "Formal definition of mapping functions Y = f(X)."
        explanation = """<h2>Formal Definition</h2><p>Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output: Y = f(X).</p>"""

    elif "data preprocessing" in t_lower:
        description = "Preparing raw data for ML models: handling missing data and encoding."
        explanation = """<h2>Garbage In, Garbage Out</h2><p>Models are only as good as the data they are fed. Preprocessing transforms raw data into a clean, understandable format.</p><ul><li>Handling Missing Data</li><li>Encoding Categorical Variables</li><li>Scaling Features</li></ul>"""

    elif "mean removal" in t_lower:
        description = "Centering data by removing the mean."
        explanation = """<h2>Standardization</h2><p>Mean removal involves subtracting the mean from each data point so that the resulting mean is 0. This centers the data.</p><p>Formula: x' = x - mean(x)</p>"""
        code_example = "from sklearn.preprocessing import scale\nscaled_data = scale(data)"

    elif "scaling & normalization" in t_lower:
        description = "Rescaling data to a fixed range or standard deviation."
        explanation = """<h2>Normalization</h2><p>Scaling data to a fixed range, usually 0 to 1.</p><h2>Standardization</h2><p>Scaling data to have a mean of 0 and standard deviation of 1. Algorithms like SVM and K-Means require this to function correctly.</p>"""
    
    elif "binarization" in t_lower:
        description = "Converting continuous features to binary and text to numbers."
        explanation = """<h2>Binarization</h2><p>Converting numerical features into binary (0 or 1) based on a threshold.</p><h2>Label Encoding</h2><p>Converting text labels (Cat, Dog) into numbers (0, 1).</p>"""
        code_example = "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ny_encoded = le.fit_transform(['cat', 'dog', 'cat'])"

    elif "linear regression – basics" in t_lower:
        description = "Modeling the relationship between variables with a linear equation."
        explanation = """<h2>The Line of Best Fit</h2><p>Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data.</p><p>Equation: y = mx + c</p>"""
        code_example = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X, y)"

    elif "linear regression – case study" in t_lower:
        description = "Practical Application: Predicting Boston Housing Prices."
        explanation = """<h2>Predicting House Prices</h2><p>In this case study, we use the Boston Housing dataset. Features include 'Number of Rooms', 'Crime Rate', etc. The target is 'Median Value'.</p><p>We split data, train the regressor, and evaluate using Mean Squared Error.</p>"""

    elif "introduction to classification" in t_lower:
         description = "Distinguishing Classification (Discrete) from Regression (Continuous)."
         explanation = """<h2>Classification vs Regression</h2><p>Regression predicts continuous values (Price, Temperature). Classification predicts discrete categories (Yes/No, Spam/Ham, Red/Blue).</p>"""

    elif "simple classifier" in t_lower:
        description = "Building a basic rule-based classifier."
        explanation = """<h2>Thresholding</h2><p>The simplest classifier is a rule-based system. Example: IF Age > 18 THEN Adult ELSE Minor.</p>"""

    elif "logistic regression classifier" in t_lower:
        description = "Using the Sigmoid function for probability-based classification."
        explanation = """<h2>Probabilistic Classifier</h2><p>Despite the name, Logistic Regression is for classification. It uses the Sigmoid function to squash outputs between 0 and 1, representing probability.</p>"""
        code_example = "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()"

    elif "naïve bayes" in t_lower:
        description = "Fast, probabilistic classification based on Bayes' Theorem."
        explanation = """<h2>Bayes' Theorem</h2><p>Naive Bayes assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. It is very fast and effective for text classification.</p>"""

    elif "training & testing dataset" in t_lower:
        description = "Splitting data to prevent overfitting: Train vs Test sets."
        explanation = """<h2>The Split</h2><p>We split data into Training (to fit the model) and Testing (to evaluate). Common tools: <code>train_test_split</code> from sklearn.</p>"""

    elif "accuracy & cross-validation" in t_lower:
        description = "Evaluating model performance and stability."
        explanation = """<h2>Accuracy</h2><p>Correct Predictions / Total Predictions.</p><h2>Cross-Validation</h2><p>Splitting data into K folds and training K times to ensure the model produces stable results on different subsets.</p>"""
    
    elif "confusion matrix" in t_lower:
        description = "Visualizing True Positives, False Positives, and errors."
        explanation = """<h2>Visualizing Performance</h2><p>A matrix showing TP, TN, FP, and FN. It gives a better idea of performance than just accuracy, especially for imbalanced datasets.</p>"""
        code_example = "from sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)"

    elif "classification performance" in t_lower:
        description = "Precision, Recall, and F1-Score metrics explained."
        explanation = """<h2>Precision, Recall, F1-Score</h2><ul><li><strong>Precision:</strong> Of all predicted positives, how many were real?</li><li><strong>Recall:</strong> Of all real positives, how many did we find?</li><li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall.</li></ul>"""
        code_example = "from sklearn.metrics import classification_report\nprint(classification_report(y_true, y_pred))"

    elif "predictive modeling" in t_lower:
        description = "Using historical data to forecast future outcomes."
        explanation = """<h2>Future Forecasting</h2><p>Predictive modeling allows us to predict future outcomes based on historical data. It is the core application of Supervised Learning.</p>"""

    elif "svm – theory" in t_lower:
        description = "Understanding Hyperplanes and Maximal Margins."
        explanation = """<h2>Maximal Margin Classifier</h2><p>SVM finds the hyperplane that separates classes with the maximum margin. The data points tied to the margin are called Support Vectors.</p>"""

    elif "linear & non-linear" in t_lower:
        description = "Using the Kernel Trick to separate complex data."
        explanation = """<h2>The Kernel Trick</h2><p>If data isn't linearly separable, SVM projects it into a higher dimension where it IS separable using Kernels (RBF, Polynomial).</p>"""

    elif "confidence measurements" in t_lower:
        description = "Interpreting distance from the decision boundary as confidence."
        explanation = """<h2>Distance from Hyperplane</h2><p>SVM can output the distance of a point from the decision boundary. The larger the distance, the more confident the classification.</p>"""

    elif "svm case study" in t_lower:
         description = "Application: Breast Cancer Tumor Classification."
         explanation = """<h2>Breast Cancer Classification</h2><p>Using SVM to classify tumors as Benign or Malignant based on cell features. SVM is often chosen for medical diagnosis due to its high accuracy in high-dimensional spaces.</p>"""

    # --- UNIT 3: UNSUPERVISED ---
    elif "introduction to unsupervised" in t_lower:
        description = "Finding hidden patterns in unlabeled data."
        explanation = """<h2>Discovering Structure</h2><p>Unsupervised learning finds hidden patterns or intrinsic structures in data. It is used for exploratory analysis and dimensionality reduction.</p>"""

    elif "clustering – overview" in t_lower:
        description = "Grouping similar data points together."
        explanation = """<h2>Grouping</h2><p>Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.</p>"""

    elif "k-means clustering – theory" in t_lower:
        description = "Iterative centroid-based clustering algorithm."
        explanation = """<h2>Centroid-Based</h2><p>K-Means partitions data into K clusters. It iteratively moves the centroids to the center of the assigned points until convergence.</p>"""

    elif "k-means clustering – case study" in t_lower:
        description = "Application: Customer Segmentation for Marketing."
        explanation = """<h2>Customer Segmentation</h2><p>Using K-Means to group customers based on Annual Income and Spending Score. This helps businesses target specific groups with ads.</p>"""
        code_example = "kmeans = KMeans(n_clusters=5)\nkmeans.fit(customer_data)"

    elif "image compression" in t_lower:
        description = "Reducing image size using color clustering."
        explanation = """<h2>Vector Quantization</h2><p>We can use K-Means to reduce the colors in an image. By clustering pixel colors into K=64 groups and replacing each pixel with its centroid color, we compress the image size.</p>"""

    elif "mean shift" in t_lower:
        description = "Density-based clustering using sliding windows."
        explanation = """<h2>Sliding Window</h2><p>Mean Shift is a non-parametric clustering algorithm that does not require defining the number of clusters. It shifts a window towards the area of maximum density.</p>"""

    elif "agglomerative" in t_lower:
        description = "Bottom-up Hierarchical Clustering using Dendrograms."
        explanation = """<h2>Hierarchical Clustering</h2><p>Bottom-up approach: Start with N clusters (each point). Merge the two closest clusters. Repeat until 1 cluster remains. Visualized using a Dendrogram.</p>"""

    elif "comparative study" in t_lower:
        description = "K-Means vs Hierarchical vs DBSCAN: When to use what."
        explanation = """<h2>K-Means vs Hierarchical</h2><p>K-Means is faster but assumes spherical clusters. Hierarchical is slower but captures hierarchy. DBSCAN is good for noise and arbitrary shapes.</p>"""
    

    # --- UNIT 4: NLP ---
    elif "introduction to nlp" in t_lower:
        description = "Intersection of AI, CS, and Linguistics."
        explanation = """<h2>Understanding Language</h2><p>NLP is the intersection of Computer Science, AI, and Linguistics. It enables computers to process and analyze large amounts of natural language data.</p>"""

    elif "text preprocessing – overview" in t_lower:
        description = "Cleaning and preparing text for analysis."
        explanation = """<h2>Cleaning Text</h2><p>Raw text is unstructured. Preprocessing converts it into a structured form that ML models can digest.</p>"""

    elif "text cleaning" in t_lower:
        description = "Tokenization, Stopwords Removal, and Cleaning."
        explanation = """<h2>Tokenization</h2><p>Splitting text into words.</p><h2>Stopwords</h2><p>Removing common words (the, is, and).</p><h2>Cleaning</h2><p>Removing punctuation, URLs, and HTML tags.</p>"""
        code_example = "from nltk.tokenize import word_tokenize\nwords = word_tokenize('Hello World')"

    elif "stemming" in t_lower:
        description = "Heuristic process of chopping suffixes."
        explanation = """<h2>Chopping Suffixes</h2><p>Stemming is a crude heuristic process that chops off the ends of words. 'Running' -> 'Run'. Fast but can produce non-words.</p>"""
        code_example = "from nltk.stem import PorterStemmer\nps = PorterStemmer()\nprint(ps.stem('running'))"

    elif "lemmatization" in t_lower:
        description = "Morphological analysis to find the base word form."
        explanation = """<h2>Dictionary Form</h2><p>Lemmatization uses a vocabulary and morphological analysis to return the base form of a word. 'Better' -> 'Good'. Slower but more accurate than stemming.</p>"""
        code_example = "from nltk.stem import WordNetLemmatizer\nlemmatizer = WordNetLemmatizer()\nprint(lemmatizer.lemmatize('better', pos='a'))"

    elif "chunking" in t_lower:
        description = "Grouping tokens into meaningful phrases (NP, VP)."
        explanation = """<h2>Shallow Parsing</h2><p>Chunking groups tokens into meaningful phrases, like Noun Phrases (NP) or Verb Phrases (VP). Useful for extracting entities.</p>"""

    elif "text vectorization" in t_lower:
        description = "Converting text to numbers: Bag of Words and TF-IDF."
        explanation = """<h2>Bag of Words (BoW)</h2><p>Counting word frequencies.</p><h2>TF-IDF</h2><p>Term Frequency-Inverse Document Frequency. Weighs down common words and highlights unique, important words.</p>"""
        code_example = "from sklearn.feature_extraction.text import TfidfVectorizer\nvectorizer = TfidfVectorizer()"

    elif "building a text classifier" in t_lower:
        description = "The complete NLP pipeline for classification."
        explanation = """<h2>Pipeline</h2><p>Text -> Preprocessing -> Vectorization -> Classifier (e.g. Naive Bayes) -> Prediction.</p>"""

    elif "nlp case study" in t_lower:
        description = "Sentiment Analysis on IMDB Movie Reviews."
        explanation = """<h2>Movie Review Sentiment</h2><p>Classifying IMDB movie reviews as Positive or Negative. We use TF-IDF features and a Logistic Regression classifier to achieve high accuracy.</p>"""

    # --- UNIT 5: COMPUTER VISION ---
    elif "introduction to computer vision" in t_lower:
        description = "How computers 'see' using pixels and matrices."
        explanation = """<h2>Pixels and Matrices</h2><p>Computer Vision enables computers to 'see' and interpret images. Images are stored as grids of pixel values (0-255).</p>"""

    elif "introduction to opencv" in t_lower:
        description = "Overview of the Open Source Computer Vision Library."
        explanation = """<h2>OpenCV</h2><p>The most popular library for CV. Used for real-time image processing, face detection, and object tracking.</p>"""
        code_example = "import cv2\nimg = cv2.imread('data.jpg', 0) # Read as grayscale"

    elif "haar cascades" in t_lower:
        description = "Rapid object detection using Viola-Jones algorithm."
        explanation = """<h2>Rapid Object Detection</h2><p>The Viola-Jones algorithm uses Haar-like features (rectangular patterns) to detect objects. It uses a Cascade of classifiers to reject negative regions quickly.</p>"""

    elif "face detection" in t_lower:
        description = "Implementing face detection with pre-trained XML classifiers."
        explanation = """<h2>Implementation</h2><p>We load a pre-trained XML classifier and pass our image to the <code>detectMultiScale</code> function.</p>"""
        code_example = "face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\nfaces = face_cascade.detectMultiScale(gray, 1.3, 5)"

    elif "eye, nose" in t_lower:
        description = "Detecting specific facial features within a face region."
        explanation = """<h2>Facial Feature Detection</h2><p>Similar to face detection, specific XML classifiers exist for eyes, nose, and mouth. These are often run *inside* the region of the detected face to save computation.</p>"""

    elif "object detection in images" in t_lower:
         description = "Detecting arbitrary objects (Cars, Pedestrians) in static images."
         explanation = """<h2>General Object Detection</h2><p>Beyond faces, we can detect cars, pedestrians, or bananas, provided we have a trained Haar Cascade or a Deep Learning model (like YOLO, though here we focus on OpenCV basics).</p>"""

    elif "object detection in videos" in t_lower:
        description = "Real-time object detection processing video frames."
        explanation = """<h2>Frame by Frame</h2><p>Video is just a sequence of images. We read the video frame by video, apply our detection logic to each frame, and display the result.</p>"""
        code_example = "cap = cv2.VideoCapture(0)\nwhile True:\n    ret, frame = cap.read()\n    # Detect and Draw..."

    elif "face tracking" in t_lower:
        description = "Tracking vs Detection: Following a face over time."
        explanation = """<h2>Tracking vs Detection</h2><p>Detection finds the object in every frame. Tracking finds it once and then follows it, which can be faster and smoother.</p>"""

    elif "pupil detection" in t_lower:
        description = "Advanced: Thresholding and Contour detection for Gaze Tracking."
        explanation = """<h2>Gaze Tracking Foundation</h2><p>By thresholding the eye region to isolate dark pixels, we can find the contours of the pupil. The center of this contour gives us the gaze direction.</p>"""

    elif "opencv case study" in t_lower:
        description = "Application: Driver Drowsiness Detection System."
        explanation = """<h2>Driver Drowsiness Detection</h2><p>A real-world app. We detect the face, then the eyes. If the eyes remain closed for X consecutive frames (Aspect Ratio < Threshold), we sound an alarm.</p>"""
    
    # Generic safety net (should rarely be hit now)
    else:
        description = f"Detailed overview of {title}."
        explanation = f"<h2>{title}</h2><p>Detailed exploration of {title}.</p>"
        key_points = ["Overview", "Details", "Practice"]

    return { "explanation": explanation, "code_example": code_example, "key_points": key_points, "description": description }

# THE EXACT SYLLABUS
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
        "2. Data Preprocessing – Overview",
        "3. Mean Removal (Standardization)",
        "4. Scaling & Normalization",
        "5. Binarization & Label Encoding",
        "6. Linear Regression – Basics",
        "7. Linear Regression – Case Study",
        "8. Introduction to Classification",
        "9. Building a Simple Classifier",
        "10. Logistic Regression Classifier",
        "11. Naïve Bayes Classifier",
        "12. Training & Testing Dataset",
        "13. Accuracy & Cross-Validation",
        "14. Confusion Matrix Visualization",
        "15. Classification Performance Report",
        "16. Predictive Modeling – Introduction",
        "17. Support Vector Machine (SVM) – Theory",
        "18. SVM – Linear & Non-Linear Classifiers",
        "19. Confidence Measurements in SVM",
        "20. SVM Case Study",
        "21. Supervised Learning – Summary"
    ],
    "Unit 3: Unsupervised Learning": [
        "1. Introduction to Unsupervised Learning",
        "2. Clustering – Overview",
        "3. K-Means Clustering – Theory",
        "4. K-Means Clustering – Case Study",
        "5. Image Compression via Vector Quantization",
        "6. Mean Shift Clustering",
        "7. Agglomerative Clustering",
        "8. Clustering – Comparative Study",
        "9. Unsupervised Learning – Summary",
        "10. Semi-Supervised Learning"
    ],
    "Unit 4: Natural Language Processing": [
        "1. Introduction to NLP",
        "2. Text Preprocessing – Overview",
        "3. Text Cleaning & Tokenization",
        "4. Stemming",
        "5. Lemmatization",
        "6. Chunking",
        "7. Text Vectorization",
        "8. Building a Text Classifier",
        "9. NLP Case Study",
        "10. NLP – Summary"
    ],
    "Unit 5: Computer Vision with OpenCV": [
        "1. Introduction to Computer Vision",
        "2. Introduction to OpenCV",
        "3. Haar Cascades – Theory",
        "4. Face Detection",
        "5. Eye, Nose, Mouth Detection",
        "6. Object Detection in Images",
        "7. Object Detection in Videos",
        "8. Face Tracking in Real-Time",
        "9. Pupil Detection (Advanced)",
        "10. OpenCV Case Study",
        "11. Computer Vision – Summary"
    ]
}

def create_tutorials():
    new_tutorials = []
    
    # 1. READ EXISTING content
    existing_data = []
    if os.path.exists(tutorials_path):
        with open(tutorials_path, 'r') as f:
            existing_data = json.load(f)
    
    # Filter out OLD ML content
    clean_data = [t for t in existing_data if t.get('technology') != "Machine Learning with Python"]
    
    # 2. GENERATE NEW ML CONTENT
    for unit_name, titles in syllabus_structure.items():
        for i, title in enumerate(titles):
            # Create ID (Remove ?, spaces, etc)
            clean_title = title.split('. ')[1].lower()
            clean_title = re.sub(r'[^a-z0-9\s-]', '', clean_title) # Remove special chars like ? and ()
            clean_title = clean_title.replace(' ', '-')
            
            tid = f"ml-{unit_name.split(':')[0].lower().replace(' ', '')}-{clean_title}"
            
            # Fetch Content - PASSING UNIT NAME NOW
            detailed_content = get_tutorial_content(title, unit_name)
            
            # Build Object
            tutorial_obj = {
                "id": tid,
                "title": title,
                "technology": "Machine Learning with Python",
                "unit": unit_name,
                "definition": detailed_content["description"], 
                "description": detailed_content["description"],
                "syntax": "Theory & Practice",
                "code_example": detailed_content["code_example"],
                "explanation": detailed_content["explanation"],
                "try_it_yourself": True,
                "key_points": detailed_content["key_points"]
            }
            new_tutorials.append(tutorial_obj)

    # 3. MERGE & WRITE
    clean_data.extend(new_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    
    print(f"Successfully created {len(new_tutorials)} tutorials with UNIQUE CONTENT.")

if __name__ == "__main__":
    create_tutorials()
