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
            {"question": "Define Machine Learning.", "answer": "The science of getting computers to act without being explicitly programmed."},
            {"question": "What is Supervised Learning?", "answer": "Learning from labeled training data."},
            {"question": "What is AI?", "answer": "Artificial Intelligence: Simulation of human intelligence by machines."},
            {"question": "List one application of ML.", "answer": "Email Spam Filtering."},
            {"question": "What is a Label?", "answer": "The output variable we want to predict."}
        ],
        "Part B (2-Marks)": [
            {"question": "Differentiate AI vs ML vs DL.", "answer": "AI is the broad field, ML is a subset using statistical methods, DL is a subset of ML using neural networks."},
            {"question": "Explain Reinforcement Learning.", "answer": "An agent learns by interacting with an environment and receiving rewards or penalties."},
            {"question": "What are the 7 steps of ML?", "answer": "Gathering, Preprocessing, Model Choice, Training, Testing, Tuning, Prediction."}
        ],
        "Part C (5-Marks)": [
            {"question": "Explain the Machine Learning Pipeline in detail.", "answer": "Detailed explanation of Data Gathering, Cleaning, Training, and Evaluation..."},
            {"question": "Compare Supervised vs Unsupervised Learning with examples.", "answer": "Supervised uses labels (Spam filter). Unsupervised finds patterns (Customer segmentation)."}
        ]
    },
    "Unit 2: Supervised Learning": {
        "Part A (1-Mark)": [
            {"question": "What is Regression?", "answer": "Predicting continuous values (e.g., Price)."},
            {"question": "What is Classification?", "answer": "Predicting discrete labels (e.g., Red/Blue)."},
            {"question": "Define Confusion Matrix.", "answer": "A table used to describe the performance of a classifier."},
            {"question": "What is Feature Scaling?", "answer": "Normalizing the range of independent variables."},
            {"question": "What is SVM?", "answer": "Support Vector Machine."}
        ],
        "Part B (2-Marks)": [
            {"question": "Explain Linear Regression.", "answer": "Modeling the relationship between scalar response and explanatory variables using a linear equation."},
            {"question": "What are Support Vectors?", "answer": "Data points closest to the hyperplane that influence its position."},
            {"question": "Explain Naive Bayes.", "answer": "A probabilistic classifier based on Bayes' Theorem assuming independence between features."}
        ],
        "Part C (5-Marks)": [
            {"question": "Explain the SVM algorithm and the Kernel Trick.", "answer": "SVM finds the optimal hyperplane. Kernel trick maps non-linear data to higher dimensions..."},
            {"question": "Describe the metrics used to evaluate a Classifier.", "answer": "Accuracy, Precision, Recall, F1-Score, and Confusion Matrix."}
        ]
    },
    "Unit 3: Unsupervised Learning": {
        "Part A (1-Mark)": [
            {"question": "Define Clustering.", "answer": "Grouping a set of objects in such a way that objects in the same group are similar."},
            {"question": "What is K-Means?", "answer": "A centroid-based clustering algorithm."},
            {"question": "What is a Dendrogram?", "answer": "A tree diagram used to illustrate the arrangement of clusters."},
            {"question": "What is unlabeled data?", "answer": "Data without output targets/labels."},
            {"question": "Define Mean Shift.", "answer": "A sliding-window centroid-based algorithm."}
        ],
        "Part B (2-Marks)": [
            {"question": "How does K-Means work?", "answer": "Initialize K centroids, assign points to nearest centroid, update centroids, repeat."},
            {"question": "What is Vector Quantization?", "answer": "A compression technique used in image processing using clustering."},
            {"question": "Differnce between K-Means and Hierarchical Clustering.", "answer": "K-Means requires K beforehand and is fast. Hierarchical builds a tree structure and is slower."}
        ],
        "Part C (5-Marks)": [
            {"question": "Explain the K-Means Clustering algorithm with an example.", "answer": "Step-by-step definition of initialization, assignment, and update steps..."},
            {"question": "Discuss Agglomerative Hierarchical Clustering.", "answer": "Bottom-up approach starting with each point as a cluster and merging them..."}
        ]
    },
    "Unit 4: Natural Language Processing": {
        "Part A (1-Mark)": [
            {"question": "What is NLP?", "answer": "Natural Language Processing."},
            {"question": "Define Tokenization.", "answer": "Splitting text into sentences or words."},
            {"question": "What is a Stopword?", "answer": "A common word (the, is) usually removed during preprocessing."},
            {"question": "What is Stemming?", "answer": "Reducing words to their root form (chopping suffixes)."},
            {"question": "What is BoW?", "answer": "Bag of Words."}
        ],
        "Part B (2-Marks)": [
            {"question": "Differentiate Stemming vs Lemmatization.", "answer": "Stemming chops words (fast, crude). Lemmatization finds dictionary root (slower, accurate)."},
            {"question": "Explain TF-IDF.", "answer": "Term Frequency - Inverse Document Frequency. Measures word importance."},
            {"question": "What is Sentiment Analysis?", "answer": "Determining the emotional tone behind a series of words."}
        ],
        "Part C (5-Marks)": [
            {"question": "Explain the text preprocessing pipeline in NLP.", "answer": "Cleaning -> Tokenization -> Stopword Removal -> Stemming/Lemmatization -> Vectorization."},
            {"question": "Describe the Bag of Words model with an example.", "answer": "Representation of text that describes the occurrence of words within a document..."}
        ]
    },
    "Unit 5: Computer Vision with OpenCV": {
        "Part A (1-Mark)": [
            {"question": "What is OpenCV?", "answer": "Open Source Computer Vision Library."},
            {"question": "What represents an image in ML?", "answer": "A matrix of pixel values."},
            {"question": "What is a Haar Cascade?", "answer": "A machine learning object detection method used to identify objects in images."},
            {"question": "Function to read an image in OpenCV?", "answer": "cv2.imread()"},
            {"question": "What is Object Detection?", "answer": "Identifying instances of objects of a certain class in images."}
        ],
        "Part B (2-Marks)": [
            {"question": "Explain the Viola-Jones algorithm.", "answer": "A framework for face detection using Haar features and Cascade classifiers."},
            {"question": "Difference between Detection and Tracking.", "answer": "Detection finds object in every frame. Tracking locates it once and follows it."},
            {"question": "How do we detect eyes?", "answer": "Using a trained Haar Cascade XML specifically for eyes, usually within the detected face region."}
        ],
        "Part C (5-Marks)": [
            {"question": "Explain Object Detection using Haar Cascades.", "answer": "Using positive/negative images to train a classifier that looks for specific features (edges, lines)..."},
            {"question": "Discuss a Case Study on Driver Drowsiness Detection.", "answer": "Detecting face -> Detecting eyes -> Measuring aspect ratio over time to detect closure."}
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
        
    print(f"Successfully generated ML Question Bank for {len(final_data)} units.")

if __name__ == "__main__":
    generate_exam_json()
