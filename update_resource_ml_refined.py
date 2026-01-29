import json
import os

# Define the ML Syllabus units
ML_UNITS = [
    "Unit 1: Introduction to ML",
    "Unit 2: Supervised Learning",
    "Unit 3: Unsupervised Learning",
    "Unit 4: Natural Language Processing",
    "Unit 5: Computer Vision with OpenCV"
]

# Define Questions for each unit
# Structure: Unit Name -> Sections -> List of Qs
QUESTION_BANK = {
    "Unit 1: Introduction to ML": {
        "Part A (1-Mark)": [
            {"question": "Define Machine Learning.", "answer": "Learning from data without explicit programming."},
            {"question": "What is Supervised Learning?", "answer": "Learning with labeled data."},
            {"question": "What is AI?", "answer": "Simulation of human intelligence by machines."},
            {"question": "What is a Label?", "answer": "The target output variable."},
            {"question": "Define Data.", "answer": "Raw facts and figures used for learning."}
        ],
        "Part B (2-Marks)": [
            {"question": "AI vs ML.", "answer": "AI is the broad science; ML is a subset that learns from data."},
            {"question": "What is Reinforcement Learning?", "answer": "Learning via rewards and penalties."},
            {"question": "Define Training Set.", "answer": "The subset of data used to train the model."},
            {"question": "Define Testing Set.", "answer": "The subset of data used to evaluate performance."},
            {"question": "What is a Feature?", "answer": "An input variable used for making predictions."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain the 3 types of Learning.", "answer": "1. Supervised (Labeled)\n2. Unsupervised (Unlabeled)\n3. Reinforcement (Reward-based)"},
            {"question": "What is Deep Learning?", "answer": "A subset of ML using neural networks to mimic the human brain."},
            {"question": "Difference between Label and Feature.", "answer": "Features are Inputs (Questions). Labels are Outputs (Answers)."}
        ],
        "Part D (5-Marks)": [
            {"question": "Explain the 7 Steps of Machine Learning.", "answer": "1. Gathering Data\n2. Preprocessing\n3. Choosing Model\n4. Training\n5. Testing\n6. Tuning\n7. Prediction"},
            {"question": "Detailed comparison of AI, ML, and DL.", "answer": "AI is the umbrella term for smart machines. ML is the subset regarding statistical learning. DL is the specialized subset using multi-layered Neural Networks."}
        ]
    },
    "Unit 2: Supervised Learning": {
        "Part A (1-Mark)": [
            {"question": "What is Regression?", "answer": "Predicting continuous values."},
            {"question": "What is Classification?", "answer": "Predicting discrete categories."},
            {"question": "Define Confusion Matrix.", "answer": "Table describing classifier performance."},
            {"question": "What is SVM?", "answer": "Support Vector Machine."},
            {"question": "What is a Hyperplane?", "answer": "The decision boundary separating classes."}
        ],
        "Part B (2-Marks)": [
            {"question": "Linear vs Logistic Regression.", "answer": "Linear predicts values (Price). Logistic predicts probability/classes (Yes/No)."},
            {"question": "What are Support Vectors?", "answer": "Data points closest to the hyperplane."},
            {"question": "Define Naive Bayes.", "answer": "Probabilistic classifier assuming feature independence."},
            {"question": "What is Overfitting?", "answer": "Model learns noise/details too well, failing on new data."},
            {"question": "What is Accuracy?", "answer": "Ratio of correct predictions to total predictions."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Precision and Recall.", "answer": "Precision: Accuracy of positive predictions.\nRecall: Ability to find all actual positives."},
            {"question": "Explain the Kernel Trick in SVM.", "answer": "Mapping non-linear data to a higher dimension to make it linearly separable."},
            {"question": "Steps in Data Preprocessing.", "answer": "1. Cleaning (Missing values)\n2. Encoding (Text to Numbers)\n3. Scaling (Normalization)"}
        ],
        "Part D (5-Marks)": [
            {"question": "Explain SVM Algorithm.", "answer": "Goal: Find hyperplane with Max Margin.\nSupport Vectors: Points defining the margin.\nKernel: Handles non-linear data."},
            {"question": "Classification Performance Metrics.", "answer": "1. Accuracy (Overall correctness)\n2. Precision (Exactness)\n3. Recall (Completeness)\n4. F1-Score (Balance)\n5. ROC Curve."}
        ]
    },
    "Unit 3: Unsupervised Learning": {
        "Part A (1-Mark)": [
            {"question": "Define Clustering.", "answer": "Grouping similar objects."},
            {"question": "What is K-Means?", "answer": "Centroid-based clustering."},
            {"question": "What is Unlabeled Data?", "answer": "Data with no target answers."},
            {"question": "What is a Centroid?", "answer": "The center point of a cluster."},
            {"question": "Define Vector Quantization.", "answer": "Image compression using clustering."}
        ],
        "Part B (2-Marks)": [
            {"question": "K-Means vs Hierarchical.", "answer": "K-Means: Fast, needs K. Hierarchical: Tree-based, slow, no K needed."},
            {"question": "What is a Dendrogram?", "answer": "Tree diagram showing hierarchical clusters."},
            {"question": "What is Mean Shift?", "answer": "Clustering by shifting window to high density."},
            {"question": "Applications of Clustering.", "answer": "Customer Segmentation, Image Compression."},
            {"question": "What is Euclidean Distance?", "answer": "Straight-line distance between two points."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Semi-Supervised Learning.", "answer": "Train model on small labeled set -> Pseudo-label the unlabeled set -> Train on combined data."},
            {"question": "Steps in K-Means.", "answer": "1. Initialize K centroids\n2. Assign points to nearest centroid\n3. Update centroids\n4. Repeat until convergence."},
            {"question": "Agglomerative vs Divisive.", "answer": "Agglomerative: Bottom-up (Merge).\nDivisive: Top-down (Split)."}
        ],
        "Part D (5-Marks)": [
            {"question": "Detailed explanation of K-Means.", "answer": "Algorithm that partitions N observations into K clusters. Minimizes inertia. Efficient but sensitive to initialization (K-Means++ fixes this)."},
            {"question": "Different Types of Clustering.", "answer": "1. Partitioning (K-Means)\n2. Hierarchical (Agglomerative)\n3. Density-Based (DBSCAN)\n4. Grid-Based."}
        ]
    },
    "Unit 4: NLP": {
        "Part A (1-Mark)": [
            {"question": "What is NLP?", "answer": "Natural Language Processing."},
            {"question": "Define Tokenization.", "answer": "Splitting text into words."},
            {"question": "What is a Stopword?", "answer": "Common word kept out (the, is)."},
            {"question": "What is Corpus?", "answer": "A large collection of text."},
            {"question": "Define Stemming.", "answer": "Chopping word suffixes."}
        ],
        "Part B (2-Marks)": [
            {"question": "Stemming vs Lemmatization.", "answer": "Stemming: Crude chopping. Lemmatization: Dictionary based root."},
            {"question": "What is BoW?", "answer": "Bag of Words: Counting word frequency."},
            {"question": "What is TF-IDF?", "answer": "Metric weighing unique words higher than common ones."},
            {"question": "What is Chunking?", "answer": "Grouping tokens into phrases (Noun Phrases)."},
            {"question": "Sentiment Analysis.", "answer": "Identifying positive/negative emotion in text."}
        ],
        "Part C (3-Marks)": [
            {"question": "NLP Pipeline Steps.", "answer": "1. Cleaning\n2. Tokenization\n3. Vectorization (BoW/TFIDF)\n4. Modeling"},
            {"question": "Explain Text Vectorization.", "answer": "Converting text to numbers. Computers need numbers. Methods: Count Vectorizer, TF-IDF."},
            {"question": "Applications of NLP.", "answer": "Chatbots, Spam Filters, Translation, Sentiment Analysis."}
        ],
        "Part D (5-Marks)": [
            {"question": "Explain TF-IDF in detail.", "answer": "TF (Term Frequency): How often word appears in doc.\nIDF (Inverse Doc Frequency): How rare word is across docs.\nScore = TF * IDF. Highlights keywords."},
            {"question": "Compare Rule-based vs ML-based NLP.", "answer": "Rule-based: Manual grammar rules (rigid). ML-based: Learns patterns from data (flexible, handles ambiguity)."}
        ]
    },
    "Unit 5: Computer Vision": {
        "Part A (1-Mark)": [
            {"question": "What is OpenCV?", "answer": "Library for Computer Vision."},
            {"question": "What is a Pixel?", "answer": "Smallest unit of an image (0-255)."},
            {"question": "Function to show image?", "answer": "cv2.imshow()"},
            {"question": "What is Grayscale?", "answer": "Image with only black/white intensity (1 channel)."},
            {"question": "What is FPS?", "answer": "Frames Per Second."}
        ],
        "Part B (2-Marks)": [
            {"question": "What affects Face Detection?", "answer": "Lighting, Pose, Occlusion, Resolution."},
            {"question": "Haar Cascade features.", "answer": "Edge features, Line features, Four-rectangle features."},
            {"question": "Detection vs Tracking.", "answer": "Detection: Every frame (slow). Tracking: Follows object (fast)."},
            {"question": "Benefits of Grayscale.", "answer": "Less data to process (1/3rd size), faster algorithms."},
            {"question": "What is Thresholding?", "answer": "Converting grayscale to binary (black/white) based on a limit."}
        ],
        "Part C (3-Marks)": [
            {"question": "Steps in Object Detection.", "answer": "1. Input Image\n2. Preprocessing (Grayscale)\n3. Feature Extraction (Haar)\n4. Classification (Object vs Background)."},
            {"question": "Explain Viola-Jones Algorithm.", "answer": "1. Haar Features\n2. Integral Image (Speed)\n3. Adaboost (Selection)\n4. Cascading (Reject negatives fast)."},
            {"question": "Applications of CV.", "answer": "Face Unlock, Self-Driving Cars, Medical Imaging."}
        ],
        "Part D (5-Marks)": [
            {"question": "Explain Drowsiness Detection System.", "answer": "1. Detect Face\n2. Detect Eyes\n3. Calculate Eye Aspect Ratio (EAR)\n4. If EAR < Threshold for Time T -> Alert."},
            {"question": "Detailed explanation of Haar Cascades.", "answer": "Machine learning approach using positive/negative images. XML file contains thousand of tiny classifiers. It checks image regions in stages (Cascade). If a region fails a stage, it's rejected instantly."}
        ]
    }
}

def generate_exam_json():
    output_path = os.path.join("backend", "data", "exam_questions.json")
    
    final_data = []
    
    for unit in ML_UNITS:
        sections = QUESTION_BANK.get(unit, {})
        final_data.append({
            "unit": unit,
            "sections": sections
        })
        
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"Successfully generated Refined ML Question Bank: 1, 2, 3, 5 Marks.")

if __name__ == "__main__":
    generate_exam_json()
