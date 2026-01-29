import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')

# Reconstructing the "Granular" Syllabus based on user logs.
# User wants "Detailed Description" (Long HTML) but "Original Topics" (Granular breakdown).

granular_tutorials = [
    # --- UNIT 1: INTRODUCTION TO ML ---
    {
        "id": "ml-unit1-intro",
        "title": "1. Introduction to Machine Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The science of getting computers to perform tasks without explicit instructions.",
        "description": "Comprehensive overview of ML, its history, and importance.",
        "explanation": """
        <h2>1. The Evolution of Intelligence</h2>
        <p>Machine Learning (ML) serves as a bridge between traditional programming and true Artificial Intelligence. In traditional programming, humans explicitly code rules (IF this THEN that). In Machine Learning, the computer <strong>learns the rules</strong> by analyzing data.</p>
        
        <h2>2. Formal Definition</h2>
        <p>Arthur Samuel (1959) described ML as: <em>"Field of study that gives computers the ability to learn without being explicitly programmed."</em></p>
        <p>Tom Mitchell (1997) provided a more mathematical definition involving Experience (E), Task (T), and Performance (P).</p>

        <h2>3. Applications in the Real World</h2>
        <ul>
            <li><strong>Healthcare:</strong> Predicting disease outbreaks and diagnosing tumors.</li>
            <li><strong>Finance:</strong> Fraud detection and algorithmic trading.</li>
            <li><strong>Marketing:</strong> Personalized recommendations (Netflix, Amazon).</li>
            <li><strong>Transportation:</strong> Self-driving cars (Tesla, Waymo).</li>
        </ul>
        """,
        "key_points": ["Arthur Samuel", "Tom Mitchell", "Data Driven"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit1-types",
        "title": "2. Types of Machine Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The three main paradigms: Supervised, Unsupervised, Reinforcement.",
        "description": "Detailed breakdown of learning categories.",
        "explanation": """
        <h2>1. Supervised Learning</h2>
        <p>This is the most common type. The model is trained on <strong>labeled data</strong>. It acts like a student learning with a teacher who provides the answer key.</p>
        <ul>
            <li><strong>Regression:</strong> Predicting continuous values (e.g., Housing Prices).</li>
            <li><strong>Classification:</strong> Predicting discrete labels (e.g., Spam vs Not Spam).</li>
        </ul>

        <h2>2. Unsupervised Learning</h2>
        <p>The model is trained on <strong>unlabeled data</strong>. It must find structure on its own, like a student trying to learn a language without a dictionary.</p>
        <ul>
            <li><strong>Clustering:</strong> Grouping similar items (e.g., Customer Segmentation).</li>
            <li><strong>Association:</strong> Finding rules (e.g., Market Basket Analysis).</li>
        </ul>

        <h2>3. Reinforcement Learning</h2>
        <p>The model learns by interacting with an environment and receiving <strong>rewards</strong> or <strong>punishments</strong>. It is widely used in Robotics and Gaming (e.g., AlphaGo).</p>
        """,
        "key_points": ["Labeled Data", "Unlabeled Data", "Rewards"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit1-workflow",
        "title": "3. The Machine Learning Lifecycle",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "From Raw Data to Deployment.",
        "description": "The standardized pipeline for building ML models.",
        "explanation": """
        <h2>1. Data Gathering</h2>
        <p>The foundation of any ML model is data. This can come from databases, APIs, or files (CSV, Excel). The quality of data determines the quality of the model (Garbage In, Garbage Out).</p>

        <h2>2. Data Preprocessing</h2>
        <p>Real-world data is messy. Steps include:</p>
        <ul>
            <li><strong>Cleaning:</strong> Handling missing values and removing duplicates.</li>
            <li><strong>Normalization:</strong> Scaling features to a similar range (0-1).</li>
            <li><strong>Encoding:</strong> Converting text labels into numbers.</li>
        </ul>

        <h2>3. Model Selection & Training</h2>
        <p>Choosing the right algorithm (e.g., Linear Regression vs Random Forest) and feeding it the training data.</p>

        <h2>4. Evaluation</h2>
        <p>Testing the model on unseen data using metrics like Accuracy, Precision, Recall, or Root Mean Squared Error (RMSE).</p>
        """,
        "key_points": ["Preprocessing", "Training", "Testing"],
        "try_it_yourself": True
    },

    # --- UNIT 2: SUPERVISED LEARNING ---
    {
        "id": "ml-unit2-intro",
        "title": "1. Introduction to Supervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Predicting outcomes based on historical data.",
        "description": "Understanding Features (X) and Labels (y).",
        "explanation": """
        <h2>1. The Concept</h2>
        <p>In Supervised Learning, we train a model using a dataset that includes both the input features (<strong>X</strong>) and the correct answers or labels (<strong>y</strong>). The goal is to learn a mapping function <code>y = f(X)</code>.</p>

        <h2>2. Train-Test Split</h2>
        <p>We cannot test the model on the same data we trained it on, or it will just memorize the answers (Overfitting). We split data into:</p>
        <ul>
            <li><strong>Training Set (70-80%):</strong> Used to learn the patterns.</li>
            <li><strong>Testing Set (20-30%):</strong> Used to validate performance.</li>
        </ul>
        """,
        "key_points": ["Inputs (X)", "Labels (y)", "Split"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-linear-theory",
        "title": "2. Linear Regression - Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Fitting the best straight line.",
        "description": "The mathematics behind Linear Regression.",
        "explanation": """
        <h2>1. The Equation of a Line</h2>
        <p>The simplest form is <code>y = mx + c</code> (or <code>y = wX + b</code> in ML notation).</p>
        <ul>
            <li><strong>y:</strong> The predicted value (Dependent Variable).</li>
            <li><strong>X:</strong> The input feature (Independent Variable).</li>
            <li><strong>w (weight):</strong> The slope or coefficient.</li>
            <li><strong>b (bias):</strong> The intercept.</li>
        </ul>

        <h2>2. The Cost Function (MSE)</h2>
        <p>How do we find the 'best' line? We calculate the error between the actual points and the line using <strong>Mean Squared Error (MSE)</strong>. The goal is to minimize this error.</p>
        """,
        "key_points": ["y=mx+c", "MSE", "Optimization"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit2-logistic",
        "title": "3. Logistic Regression (Classification)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Supervised Learning",
        "definition": "Predicting Categories.",
        "description": "From continuous lines to probability curves.",
        "explanation": """
        <h2>1. Why not Linear Regression?</h2>
        <p>Linear regression outputs continuous numbers (e.g., 105.4, -20.5). For classification (Yes/No), we need outputs between 0 and 1 representing probability.</p>

        <h2>2. The Sigmoid Function</h2>
        <p>Logistic Regression wraps the linear equation in a Sigmoid activation function:</p>
        <p>$$ S(x) = \\frac{1}{1 + e^{-x}} $$</p>
        <p>This transforms any number into a value between 0 and 1.</p>

        <h2>3. Decision Boundary</h2>
        <p>If the output > 0.5, we classify as Class 1. If < 0.5, Class 0.</p>
        """,
        "key_points": ["Sigmoid", "Probability", "0 or 1"],
        "try_it_yourself": True
    },

    # --- UNIT 3: UNSUPERVISED LEARNING ---
    {
        "id": "ml-unit3-intro",
        "title": "1. Introduction to Unsupervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Finding hidden patterns in unlabeled data.",
        "description": "Learning without a teacher.",
        "explanation": """
        <h2>1. Overview</h2>
        <p>Unsupervised learning deals with data that has no historical labels. The algorithm is left to its own devises to discover interesting structures in the data.</p>
        
        <h2>2. Key Use Cases</h2>
        <ul>
            <li><strong>Customer Segmentation:</strong> Grouping customers by purchasing behavior.</li>
            <li><strong>Anomaly Detection:</strong> Finding credit card fraud.</li>
            <li><strong>Recommendation Systems:</strong> Suggesting similar products.</li>
        </ul>
        """,
        "key_points": ["No Labels", "Discovery", "Structure"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-clustering-overview",
        "title": "2. Clustering - Overview",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Grouping similar data points.",
        "description": "Concepts of similarity and distance.",
        "explanation": """
        <h2>1. What is Clustering?</h2>
        <p>Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points as the group.</p>

        <h2>2. Distance Metrics</h2>
        <p>How do we measure 'similarity'? Usually by distance:</p>
        <ul>
            <li><strong>Euclidean Distance:</strong> The straight line distance between two points.</li>
            <li><strong>Manhattan Distance:</strong> The distance based on a grid path (like a taxi in a city).</li>
        </ul>
        """,
        "key_points": ["Similarity", "Euclidean", "Manhattan"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-kmeans-theory",
        "title": "3. K-Means Clustering - Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Centroid-based partition algorithm.",
        "description": "How the K-Means algorithm works step-by-step.",
        "explanation": """
        <h2>1. The Algorithm Steps</h2>
        <ol>
            <li><strong>Initialization:</strong> Choose K random points as the initial centroids.</li>
            <li><strong>Assignment:</strong> Assign each data point to the closest centroid.</li>
            <li><strong>Update:</strong> Calculate the new mean (centroid) of each cluster.</li>
            <li><strong>Repeat:</strong> Repeat steps 2 and 3 until the centroids stop moving (Convergence).</li>
        </ol>

        <h2>2. Choosing K (Elbow Method)</h2>
        <p>One challenge is we must tell the algorithm how many clusters (K) to find. We use the Elbow Method, plotting the Sum of Square Errors (WCSS) and picking the 'elbow' point where the drop slows down.</p>
        """,
        "key_points": ["Centroids", "Convergence", "Elbow Method"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit3-kmeans-case",
        "title": "4. K-Means Clustering - Case Study",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Unsupervised Learning",
        "definition": "Implementing K-Means in Python.",
        "description": "A practical example using Scikit-Learn.",
        "explanation": """
        <h2>1. Implementation</h2>
        <pre>
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Create Model
kmeans = KMeans(n_clusters=3)

# 2. Fit Model
kmeans.fit(data)

# 3. Get Results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Visualize
plt.scatter(data[:,0], data[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x')
plt.show()
        </pre>
        """,
        "key_points": ["sklearn", "fit()", "cluster_centers_"],
        "try_it_yourself": True
    },

    # --- UNIT 4: NLP ---
    {
        "id": "ml-unit4-intro",
        "title": "1. Introduction to NLP",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Computers understanding human language.",
        "description": "Challenges and applications of NLP.",
        "explanation": """
        <h2>1. What is NLP?</h2>
        <p>Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It draws from many disciplines, including computer science and computational linguistics.</p>

        <h2>2. Hard Challenges</h2>
        <p>Language is messy. Challenges include:</p>
        <ul>
            <li><strong>Ambiguity:</strong> "I saw the man on the hill with a telescope." (Who has the telescope?)</li>
            <li><strong>Sarcasm:</strong> "Oh, great!" (Could differ by tone).</li>
            <li><strong>Slang/Neologisms:</strong> New words are created constantly.</li>
        </ul>
        """,
        "key_points": ["Linguistics", "Ambiguity", "Context"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-preprocessing",
        "title": "2. Text Preprocessing Checklist",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Cleaning raw text.",
        "description": "Tokenization, Stopwords, and Normalization.",
        "explanation": """
        <h2>1. Cleaning Constraints</h2>
        <p>Before analysis, text must be standard. This involves:</p>
        <ul>
            <li><strong>Lowercasing:</strong> 'Apple' == 'apple'.</li>
            <li><strong>Removing Punctuation:</strong> periods, commas, etc.</li>
        </ul>

        <h2>2. Tokenization</h2>
        <p>Splitting sentences into individual words (Tokens).</p>

        <h2>3. Stopwords Removal</h2>
        <p>Removing extremely common words (the, a, an, in, is) that carry little semantic meaning.</p>

        <h2>4. Stemming & Lemmatization</h2>
        <p>Reducing words to their root. Stemming chops off ends (Running -> Run), Lemmatization uses a dictionary (Better -> Good).</p>
        """,
        "key_points": ["Tokens", "Stopwords", "Stemming"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-bow",
        "title": "3. Bag of Words (BoW)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Converting text to vectors.",
        "description": "The simplest feature extraction technique.",
        "explanation": """
        <h2>1. The Concept</h2>
        <p>Machine Learning models cannot understand text; they only understand numbers. Bag of Words converts text into a fixed-length vector of numbers by counting word occurrences.</p>

        <h2>2. How it Works</h2>
        <p>1. Create a Vocabulary of all unique words.<br>2. For each document, count how many times each word appears.</p>

        <h2>3. Limitations</h2>
        <p>It ignores grammar and word order ('Dog bit Man' vs 'Man bit Dog' have same BoW vector).</p>
        """,
        "key_points": ["Vectorization", "Frequency", "Vocabulary"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit4-app",
        "title": "4. Application: Sentiment Analysis",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Natural Language Processing",
        "definition": "Classifying opinions.",
        "description": "Determining Positive, Negative, or Neutral sentiment.",
        "explanation": """
        <h2>1. Overview</h2>
        <p>Sentiment Analysis determines the emotional tone behind a body of text. It is widely used to monitor brand reputation on social media.</p>

        <h2>2. Using TextBlob (Python)</h2>
        <pre>
from textblob import TextBlob

text = "I love this product, it is amazing!"
blob = TextBlob(text)

print(blob.sentiment.polarity) 
# Output: 0.8 (Positive)
        </pre>
        <p>A polarity > 0 is positive, < 0 is negative.</p>
        """,
        "key_points": ["TextBlob", "Polarity", "Opinion Mining"],
        "try_it_yourself": True
    },

    # --- UNIT 5: COMPUTER VISION ---
    {
        "id": "ml-unit5-intro",
        "title": "1. Intro to Computer Vision",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Visual perception for computers.",
        "description": "How computers 'see' images.",
        "explanation": """
        <h2>1. What is an Image?</h2>
        <p>To a computer, an image is just a massive grid (matrix) of numbers. Each number represents a <strong>pixel</strong> intentity.</p>

        <h2>2. Color Spaces</h2>
        <ul>
            <li><strong>Grayscale:</strong> 1 channel (0=Black, 255=White). Efficient for processing.</li>
            <li><strong>RGB:</strong> 3 channels (Red, Green, Blue). How screens display images.</li>
        </ul>

        <h2>3. OpenCV</h2>
        <p>OpenCV (Open Source Computer Vision Library) is the industry standard library for real-time computer vision.</p>
        """,
        "key_points": ["Matrix", "Pixels", "Channels"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-basics",
        "title": "2. OpenCV Basics",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Reading and displaying images.",
        "description": "Essential functions in cv2.",
        "explanation": """
        <h2>1. Loading Images</h2>
        <p><code>img = cv2.imread('path.jpg')</code> loads the image into a NumPy array.</p>

        <h2>2. Displaying Images</h2>
        <p><code>cv2.imshow('Window Name', img)</code> pops up a window.</p>

        <h2>3. Converting Color</h2>
        <p><code>gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)</code> converts color to grayscale, which is often the first step in detection pipelines.</p>
        """,
        "key_points": ["imread", "imshow", "cvtColor"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-face",
        "title": "3. Face Detection Theory",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Viola-Jones Algorithm.",
        "description": "Understanding Haar Cascades.",
        "explanation": """
        <h2>1. Haar Features</h2>
        <p>The algorithm looks for simple rectangular features (like the eye region is darker than the cheek region). It scans the whole image multiple times.</p>

        <h2>2. Integral Image</h2>
        <p>A clever mathematical trick that allows these features to be calculated incredibly fast.</p>

        <h2>3. Cascading</h2>
        <p>It's called a 'Cascade' because it consists of stages. A region must pass Stage 1 (simple check) to go to Stage 2. If it fails, it's rejected immediately. This makes it efficient enough for real-time video.</p>
        """,
        "key_points": ["Real-time", "Features", "Integral Image"],
        "try_it_yourself": True
    },
    {
        "id": "ml-unit5-face-code",
        "title": "4. Face Detection Code",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Computer Vision with OpenCV",
        "definition": "Implementation in Python.",
        "description": "Detecting faces in 10 lines of code.",
        "explanation": """
        <h2>1. The Code</h2>
        <pre>
import cv2

# 1. Load Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Read Image
img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Detect
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 4. Draw Rectangle
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
        </pre>
        """,
        "key_points": ["CascadeClassifier", "detectMultiScale", "rectangle"],
        "try_it_yourself": True
    }
]

def restore_tutorials():
    print("Writing Granular ML Syllabus to tutorials.json...")
    with open(tutorials_path, 'r') as f:
        data = json.load(f)
    
    # Clean out OLD ML content
    clean_data = [
        t for t in data 
        if t.get('technology') != "Machine Learning with Python"
    ]
    
    # Add the Granular ML Syllabus
    clean_data.extend(granular_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print("Success. Added", len(granular_tutorials), "Granular ML tutorials.")

if __name__ == "__main__":
    restore_tutorials()
