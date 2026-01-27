import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/exam_questions.json'

NEW_UNITS = [
    {
        "unit": "Unit 1: Introduction to ML",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define Machine Learning.", "answer": "ML is the science of getting computers to act without being explicitly programmed."},
                {"question": "Who defined ML formally?", "answer": "Tom Mitchell provided the formal E (Experience), T (Task), P (Performance) definition."},
                {"question": "What is AI?", "answer": "Artificial Intelligence is the broad concept of machines being able to carry out tasks in a way that we would consider 'smart'."},
                {"question": "What is Deep Learning?", "answer": "A subset of ML based on artificial neural networks with representation learning."},
                {"question": "Define Supervised Learning.", "answer": "Learning where the model performs a task under the guidance of labels (Teacher)."},
                {"question": "Define Unsupervised Learning.", "answer": "Learning where the model finds patterns in unlabeled data (No Teacher)."},
                {"question": "Define Reinforcement Learning.", "answer": "Learning by interacting with an environment and receiving rewards or punishments."},
                {"question": "What is Training Data?", "answer": "The subset of data used to train the model (usually 70-80%)."},
                {"question": "What is Testing Data?", "answer": "The subset of data used to evaluate the model's performance (usually 20-30%)."},
                {"question": "Differentiate AI vs ML.", "answer": "AI is the umbrella term for smart machines; ML is the specific subset of algorithms that learn from data."}
            ],
            "Part B (5-Marks)": [
                {"question": "Explain the 7 steps of Machine Learning.", "answer": "1. Data Collection\n2. Data Preprocessing\n3. Choose Model\n4. Training\n5. Evaluation\n6. Hyperparameter Tuning\n7. Prediction"},
                {"question": "Difference between Traditional Programming and Machine Learning.", "answer": "Traditional: Rules + Data = Answers.\nML: Data + Answers = Rules.\nIn Traditional, you code the logic. In ML, the machine learns the logic from examples."},
                {"question": "Explain Tom Mitchell's definition of ML.", "answer": "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."},
                {"question": "Compare Supervised and Unsupervised Learning.", "answer": "Supervised:\n- Labeled Data\n- Prediction/Classification\n- Example: Spam Filter\n\nUnsupervised:\n- Unlabeled Data\n- Clustering/association\n- Example: Customer Segmentation"}
            ],
            "Part C (10-Marks)": [
                {"question": "Discuss the Hierarchy of AI, ML, and DL with a diagram description.", "answer": "1. Artificial Intelligence (AI): The broadest circle. Any technique that enables, mimics human intelligence.\n2. Machine Learning (ML): A circle inside AI. Statistical techniques that give computers the ability to learn without explicit programming.\n3. Deep Learning (DL): A circle inside ML. Algorithms inspired by the structure and function of the brain called artificial neural networks.\nFrom Outer to Inner: AI -> ML -> DL."}
            ]
        }
    },
    {
        "unit": "Unit 2: Supervised Learning",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Regression?", "answer": "Predicting a continuous numerical value (e.g., Price)."},
                {"question": "What is Classification?", "answer": "Predicting a discrete label or category (e.g., Yes/No)."},
                {"question": "What is a Feature?", "answer": "An individual measurable property or characteristic of a phenomenon being observed (Input)."},
                {"question": "What is a Label?", "answer": "The result or output variable we are trying to predict."},
                {"question": "Define Overfitting.", "answer": "When a model learns the training data too well, including noise, and fails on new data."},
                {"question": "Define Underfitting.", "answer": "When a model is too simple to capture the underlying structure of the data."},
                {"question": "What is Scikit-Learn?", "answer": "A popular Python library for machine learning providing simple and efficient tools."},
                {"question": "Equation of Linear Regression?", "answer": "y = mx + c (Slope-Intercept form)."},
                {"question": "What is a Confusion Matrix?", "answer": "A table used to describe the performance of a classification model (TP, FP, TN, FN)."},
                {"question": "What is Cross-Validation?", "answer": "A technique to test the model's ability to predict new data by splitting data into K folds."}
            ],
            "Part B (5-Marks)": [
                {"question": "Explain Linear Regression Assumptions.", "answer": "1. Linearity: The relationship between X and Y is linear.\n2. Homoscedasticity: The variance of residual is the same for any value of X.\n3. Independence: Observations are independent of each other.\n4. Normality: The data (errors) follows a normal distribution."},
                {"question": "Differentiate between Regression and Classification.", "answer": "Regression:\n- Output: Continuous quantity (Numbers)\n- Algorithms: Linear Regression, SVR\n- Metrics: RMSE, MAE\n\nClassification:\n- Output: Discrete Class (Labels)\n- Algorithms: Logistic Regression, KNN, SVM\n- Metrics: Accuracy, Precision, Recall"},
                {"question": "What is the purpose of Splitting Data into Train and Test sets?", "answer": "To evaluate the model's performance on unseen data. If we test on the training data, the model might just memorize it (Overfitting). The Test set acts as a proxy for 'Real World' data to measure generalization."},
                {"question": "Explain Precision and Recall.", "answer": "Precision: Of all positive predictions, how many were actually positive? (Quality).\nRecall: Of all actual positives, how many did we find? (Quantity)."}
            ],
            "Part C (10-Marks)": [
                {"question": "Explain Support Vector Machine (SVM) and the concept of Hyperplane and Margin.", "answer": "1. SVM:\nA supervised learning algorithm used for classification and regression.\n2. Goal:\nTo find a hyperplane in an N-dimensional space that distinctly classifies the data points.\n3. Hyperplane:\nDecision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes.\n4. Margin:\nThe distance between the hyperplane and the nearest data point from either class. A 'Good' margin is one where this separation is larger. SVM maximizes this margin.\n5. Support Vectors:\nData points that are closer to the hyperplane and influence the position and orientation of the hyperplane."}
            ]
        }
    },
    {
        "unit": "Unit 3: Unsupervised Learning",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Clustering?", "answer": "The task of grouping a set of objects in such a way that objects in the same group are more similar."},
                {"question": "What is K-Means?", "answer": "A centroid-based clustering algorithm that partitions data into K clusters."},
                {"question": "What is a Centroid?", "answer": "The center point of a cluster."},
                {"question": "Define Dimensionality Reduction.", "answer": "Reducing the number of random variables under consideration (Features)."},
                {"question": "What is the Elbow Method?", "answer": "A heuristic used to determine the optimal number of clusters K."},
                {"question": "What is Euclidean Distance?", "answer": "The straight-line distance between two points in a space."},
                {"question": "What is Mean Shift?", "answer": "A non-parametric clustering algorithm that does not require defining the number of clusters."},
                {"question": "What is Agglomerative Clustering?", "answer": "A bottom-up hierarchical clustering method."},
                {"question": "What is a Dendrogram?", "answer": "A diagram that shows the hierarchical relationship between objects."},
                {"question": "What is K-Modes?", "answer": "A variation of K-Means used for categorical data."}
            ],
            "Part B (5-Marks)": [
                {"question": "Explain the K-Means Algorithm steps.", "answer": "1. Choose the number of clusters K.\n2. Select K random points as centroids.\n3. Assign each data point to the closest centroid.\n4. Compute the new centroid of each cluster.\n5. Reassign each data point to the new closest centroid.\n6. Repeat steps 4-5 until centroids do not change (Convergence)."},
                {"question": "Compare K-Means vs Hierarchical Clustering.", "answer": "K-Means:\n- Need to specify K.\n- Faster (O(n)).\n- Good for large datasets.\n\nHierarchical:\n- No need to specify K (Dendrogram).\n- Slower (O(n^3)).\n- Calculating the tree structure is heavy."},
                {"question": "What is Image Compression using Clustering?", "answer": "Using K-Means (Vector Quantization) to reduce the number of colors in an image. If an image has 16 million colors, we can cluster them into K=16 buckets and replace every pixel with its cluster center, drastically reducing size."},
                {"question": "Explain the Mean Shift algorithm.", "answer": "It involves shifting a data point iteratively to the average (mean) of data points in its neighborhood. It climbs the density gradient to find the 'peak' (mode) of the data distribution. No K required."}
            ],
            "Part C (10-Marks)": [
                {"question": "Explain the difference between Supervised and Unsupervised Learning with detailed examples.", "answer": "Supervised Learning:\n- 'Learning with a Teacher'.\n- Input labels are provided.\n- Goal: Predict outcome.\n- Example: Weather Prediction (Regression). We have historical data (Temp, Humidity) AND the result (Rain/No Rain). The model learns the mapping.\n\nUnsupervised Learning:\n- 'Learning without a Teacher'.\n- No explicit labels.\n- Goal: Find hidden structure.\n- Example: Market Basket Analysis. We have transaction data, but no target. We want to find patterns like 'People who buy Bread also buy Butter' (Association/Clustering)."}
            ]
        }
    },
    {
        "unit": "Unit 4: Natural Language Processing",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "Define NLP.", "answer": "A field of AI that gives machines the ability to read, understand and derive meaning from human languages."},
                {"question": "What is Tokenization?", "answer": "The process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements (tokens)."},
                {"question": "What are Stopwords?", "answer": "Common words (is, the, at) which are filtered out before processing natural language data."},
                {"question": "What is Stemming?", "answer": "The process of reducing inflected words to their word stem (base or root form)."},
                {"question": "What is Lemmatization?", "answer": "Grouping continuous forms of a word so they can be analyzed as a single item, based on dictionary definition."},
                {"question": "What is Corpus?", "answer": "A large and structured set of texts used for statistical analysis."},
                {"question": "What is Bag of Words?", "answer": "A representation of text that describes the occurrence of words within a document."},
                {"question": "What is TF-IDF?", "answer": "Term Frequency-Inverse Document Frequency. A statistic to evaluate how important a word is to a document."},
                {"question": "What is NLTK?", "answer": "Natural Language Toolkit. A leading platform for building Python programs to work with human language data."},
                {"question": "What is Sentiment Analysis?", "answer": "Identifying and categorizing opinions expressed in text (Positive/Negative/Neutral)."}
            ],
            "Part B (5-Marks)": [
                {"question": "Stemming vs Lemmatization. Which is better?", "answer": "Stemming:\n- Just chops off the end of the word.\n- Fast.\n- Result might not be a real word (e.g., 'Caring' -> 'Car').\n\nLemmatization:\n- Uses a dictionary/morphological analysis.\n- Slower.\n- Result is always a real word (e.g., 'Caring' -> 'Care').\n\nLemmatization is generally 'better' for accuracy; Stemming is better for speed."},
                {"question": "Explain the Bag of Words model.", "answer": "BoW is a simplifying representation used in NLP. In this model, a text (sentence or document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. It converts text into a fixed-length vector of counts."},
                {"question": "What are the steps in Text Preprocessing?", "answer": "1. Cleaning (HTML tags, URLs removal)\n2. Lowercasing\n3. Tokenization\n4. Stopword Removal\n5. Stemming/Lemmatization\n6. Vectorization (BoW/TF-IDF)"}
            ],
            "Part C (10-Marks)": [
                {"question": "Explain TF-IDF in detail.", "answer": "1. Term Frequency (TF):\nMeasures how frequently a term occurs in a document. TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).\n\n2. Inverse Document Frequency (IDF):\nMeasures how important a term is. While computing TF, all terms are considered equally important. However, certain terms, such as \"is\", \"of\", and \"that\", may appear a lot but have little importance. IDF(t) = log_e(Total number of documents / Number of documents with term t in it).\n\n3. TF-IDF Score:\nTF-IDF = TF * IDF.\nA high weight in TF-IDF is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents."}
            ]
        }
    },
    {
        "unit": "Unit 5: Computer Vision with OpenCV",
        "sections": {
            "Part A (1-Mark)": [
                {"question": "What is Computer Vision?", "answer": "A field of AI that trains computers to interpret and understand the visual world."},
                {"question": "What is OpenCV?", "answer": "Open Source Computer Vision Library. A huge open-source library for computer vision, ML, and image processing."},
                {"question": "What are Pixels?", "answer": "The smallest unit of a digital image or graphic that can be displayed."},
                {"question": "What is Grayscale?", "answer": "An image in which the value of each pixel is a single sample, representing only an amount of light (Intensity)."},
                {"question": "What is Haar Cascade?", "answer": "A machine learning object detection method used to identify objects in images or video."},
                {"question": "What is the Viola-Jones Algorithm?", "answer": "An object detection framework which provides competitive object detection rates in real-time (basis of Haar Cascades)."},
                {"question": "What is ROI?", "answer": "Region of Interest. A selected subset of samples within a dataset identified for a particular purpose."},
                {"question": "What is a Bounding Box?", "answer": "A rectangular box that encloses an object in an image."},
                {"question": "Function to read an image in OpenCV?", "answer": "cv2.imread()"},
                {"question": "Function to show an image in OpenCV?", "answer": "cv2.imshow()"}
            ],
            "Part B (5-Marks)": [
                {"question": "Explain how Face Detection works using Haar Cascades.", "answer": "1. Haar Feature Selection: Features like edge features, line features, four-rectangle features.\n2. Inegral Image: Creates an integral image for rapid computation of Haar features.\n3. Adaboost Training: Selects the best features and trains the classifiers.\n4. Cascading Classifiers: Features are grouped into stages. If a region fails the first stage, it is discarded (Fast). If it passes, it goes to the second stage."},
                {"question": "How do you capture video from a webcam in OpenCV?", "answer": "1. Create a VideoCapture object: cap = cv2.VideoCapture(0).\n2. Loop indefinitely: while(True).\n3. Read frame: ret, frame = cap.read().\n4. Show frame: cv2.imshow('frame', frame).\n5. Check for exit key: if cv2.waitKey(1) == ord('q'): break.\n6. Release: cap.release()."},
                {"question": "Why do we convert images to Grayscale for detection?", "answer": "1. Reduces Complexity: Color (3 channels) vs Gray (1 channel). 3x less data.\n2. Computational Speed: Detection algorithms work MUCH faster.\n3. Robustness: Luminance is more important than color for detecting shapes/features."}
            ],
            "Part C (10-Marks)": [
                {"question": "Explain the concept of Object Tracking.", "answer": "Object tracking is the process of locating a moving object (or multiple objects) over time using a camera. It has two main components:\n1. Detection: Identifying the object in the frame (e.g., using Haar Cascades or YOLO).\n2. Tracking: Following the object across frames.\n\nTechniques:\n- Meanshift/Camshift: Density-based tracking.\n- Optical Flow: calculating the motion vectors of pixels.\n\nCase Study - Drowsiness Detection:\n1. Detect Face.\n2. Detect Eyes within Face.\n3. Track Eyes.\n4. If Eyes remain closed (missing pupil) for N frames, Sound Alarm."}
            ]
        }
    }
]

def update_questions():
    if not os.path.exists(JSON_PATH):
        print("Error: exam_questions.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Remove existing ML units to avoid duplicates
    cleaned_data = []
    ml_unit_names = [u["unit"] for u in NEW_UNITS]
    
    for unit in data:
        # If it's not one of our new ML units, keep it
        # Also remove old legacy names if any
        if unit.get("unit") not in ml_unit_names:
            if "Unit 4: NLP" not in unit.get("unit") and "Unit 5: Computer Vision" not in unit.get("unit"):
                cleaned_data.append(unit)

    # Append new units
    cleaned_data.extend(NEW_UNITS)

    with open(JSON_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Successfully added comprehensive Exam Questions for ML Units 1-5.")

if __name__ == "__main__":
    update_questions()
