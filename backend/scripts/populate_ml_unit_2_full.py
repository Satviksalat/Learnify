
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR ML UNIT 2
ML_UNIT_2_DATA = {
    "unit": "ML Unit 2: Supervised Learning",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Regression.", "answer": "A supervised learning task where the model predicts continuous numerical values."},
            {"question": "Define Classification.", "answer": "A supervised learning task where the model predicts categorical labels or classes."},
            {"question": "What is a Hyperplane?", "answer": "A decision boundary that separates different classes in a high-dimensional space."},
            {"question": "Define Support Vectors.", "answer": "The data points closest to the hyperplane that influence its position and orientation."},
            {"question": "What is a Confusion Matrix?", "answer": "A table used to evaluate the performance of a classification model (TP, TN, FP, FN)."},
            {"question": "Define Accuracy.", "answer": "The ratio of correctly predicted observations to the total observations."},
            {"question": "What is Overfitting?", "answer": "When a model learns the training data too well, including noise, and fails on new data."},
            {"question": "What is Underfitting?", "answer": "When a model is too simple to capture the underlying structure of the data."},
            {"question": "What is the assumption of Naïve Bayes?", "answer": "It assumes that all features are independent of each other (Class Conditional Independence)."},
            {"question": "What is the Sigmoid Function?", "answer": "An activation function used in Logistic Regression to map predictions to probabilities (0 to 1)."},
            {"question": "Define Label Encoding.", "answer": "Converting categorical text labels into unique integers (e.g., Cat=0, Dog=1)."},
            {"question": "What is Normalization?", "answer": "Scaling data to fall within a specific range, typically 0 to 1."},
            {"question": "Define Binarization.", "answer": "The process of converting numerical features into binary (0 or 1) values based on a threshold."},
            {"question": "What is a Decision Tree?", "answer": "A flowchart-like model that makes decisions by splitting data based on feature values."},
            {"question": "Define Entropy in ML.", "answer": "A measure of impurity or randomness in a dataset used in Decision Trees."},
            {"question": "What is Information Gain?", "answer": "The reduction in entropy achieved by splitting the data on a specific feature."},
            {"question": "Define K-Nearest Neighbors (KNN).", "answer": "A lazy learning algorithm that classifies a point based on the majority class of its 'K' neighbors."},
            {"question": "What is 'Pruning' in Decision Trees?", "answer": "The process of removing parts of the tree that do not provide power to classify instances (reduces overfitting)."},
            {"question": "Define Mean Squared Error (MSE).", "answer": "The average of the squared differences between the predicted and actual values."},
            {"question": "What is the Kernel Trick?", "answer": "A technique in SVM to transform data into higher dimensions to find a linear separator."}
        ],
        "Part B (2-Marks)": [
            {"question": "Difference between Linear and Logistic Regression.", "answer": "• Linear: Predicts continuous values (e.g., Price). Uses best fit line.\n• Logistic: Predicts probabilities/classes (e.g., Yes/No). Uses Sigmoid curve."},
            {"question": "Parametric vs Non-Parametric Models.", "answer": "• Parametric: Fixed number of parameters (e.g., Linear Regression). Faster.\n• Non-Parametric: Flexible number of parameters (e.g., KNN, Decision Trees). More complex."},
            {"question": "Explain Bias vs Variance.", "answer": "• Bias: Error due to overly simple assumptions (leads to Underfitting).\n• Variance: Error due to sensitivity to small fluctuations (leads to Overfitting)."},
            {"question": "Standardization vs Normalization.", "answer": "• Standardization: Centers data around mean with unit variance (Z-score).\n• Normalization: Scales data to a fixed range [0, 1] (Min-Max Scaling)."},
            {"question": "What is K-Fold Cross-Validation?", "answer": "• Splitting data into 'K' subsets (folds).\n• Training on K-1 folds and testing on the remaining fold, repeating K times."},
            {"question": "What is Grid Search?", "answer": "• An exhaustive search over a specified parameter values for an estimator.\n• Used to find the optimal hyperparameters for a model."},
            {"question": "Bagging vs Boosting.", "answer": "• Bagging: Trains models in parallel independent of each other (e.g., Random Forest).\n• Boosting: Trains models sequentially, each correcting the previous one's errors."},
            {"question": "Define Random Forest.", "answer": "• An ensemble method using multiple Decision Trees.\n• Output is the majority vote (Classification) or average (Regression) of trees."},
            {"question": "Goal of Support Vector Machine (SVM).", "answer": "• To find the optimal hyperplane that maximizes the margin between classes.\n• Works well for both linear and non-linear classification."},
            {"question": "What is RMSE?", "answer": "• Root Mean Squared Error.\n• Square root of the average squared errors; keeps units same as target variable."},
            {"question": "Precision vs Recall.", "answer": "• Precision: Out of all predicted positives, how many were actually positive?\n• Recall: Out of all actual positives, how many were correctly predicted?"},
            {"question": "What is a 'Lazy Learner'?", "answer": "• An algorithm that does not build a model during training.\n• It simply stores data and computes during prediction (e.g., KNN)."},
            {"question": "Role of 'K' in KNN.", "answer": "• K is the number of nearest neighbors to look at.\n• Small K = Noise sensitive. Large K = Smoother boundaries."},
            {"question": "What is Gradient Descent?", "answer": "• An optimization algorithm to minimize the loss function.\n• Iteratively moves towards the minimum error by updating weights."},
            {"question": "What is the R-Squared metric?", "answer": "• Statistical measure of how close the data are to the fitted regression line.\n• 1.0 means perfect fit, 0.0 means no fit."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Simple Linear Regression.", "answer": "1. **Definition:** A method to predict a dependent variable (Y) based on a single independent variable (X).\n2. **Explanation:** Equation: Y = mX + c. It finds the 'Line of Best Fit' minimizing error.\n3. **Example:** Predicting Weight based on Height."},
            {"question": "Explain Logistic Regression.", "answer": "1. **Definition:** A classification algorithm for binary outcomes (0 or 1).\n2. **Explanation:** Uses the Sigmoid function 1/(1+e^-z) to squash output between 0 and 1.\n3. **Example:** Predicting if a customer will Churn (Yes/No)."},
            {"question": "Explain K-Nearest Neighbors (KNN).", "answer": "1. **Definition:** A simple, instance-based learning algorithm.\n2. **Explanation:** Calculates distance (Euclidean) between new point and existing points. Assigns most common class among K neighbors.\n3. **Example:** Recommending similar products based on user history."},
            {"question": "Explain Decision Tree Construction.", "answer": "1. **Definition:** Splitting data recursively based on features to create a tree structure.\n2. **Explanation:** Uses metrics like Gini Impurity or Entropy to choose the best split at each node.\n3. **Example:** Approved for Loan? (Income > 50k? -> Credit Score > 700? -> Yes)."},
            {"question": "Explain Random Forest.", "answer": "1. **Definition:** An ensemble technique used for classification and regression.\n2. **Explanation:** Builds many Decision Trees on random subsets of data and averages their results.\n3. **Example:** Diagnosing a disease using opinions from 100 different doctors (trees)."},
            {"question": "SVM Margins: Soft vs Hard.", "answer": "1. **Definition:** How strict the SVM is about misclassification.\n2. **Explanation:** Hard Margin: No errors allowed (prone to overfitting). Soft Margin: Some errors allowed (better generalization).\n3. **Example:** Soft margin is better for noisy real-world data."},
            {"question": "Naive Bayes Types.", "answer": "1. **Definition:** Variants of Naive Bayes based on data distribution.\n2. **Explanation:** Gaussian (Continuous data), Multinomial (Word counts), Bernoulli (Binary features).\n3. **Example:** Multinomial NB is standard for Text Classification."},
            {"question": "Confusion Matrix Elements.", "answer": "1. **Definition:** The four outcomes of binary classification.\n2. **Explanation:** TP (True Positive), TN (True Negative), FP (False Positive - Type 1 Error), FN (False Negative - Type 2 Error).\n3. **Example:** FP: Healthy person diagnosed as Sick. FN: Sick person diagnosed as Healthy."},
            {"question": "Explain ROC Curve.", "answer": "1. **Definition:** Receiver Operating Characteristic curve.\n2. **Explanation:** Plots TPR (Recall) vs FPR (1-Specificity) at various thresholds. Area Under Curve (AUC) measures performance.\n3. **Example:** AUC=0.9 is excellent, AUC=0.5 is random guessing."},
            {"question": "Bias-Variance Tradeoff.", "answer": "1. **Definition:** The balance required to minimize total error.\n2. **Explanation:** High Bias = Underfitting. High Variance = Overfitting. We want low bias and low variance.\n3. **Example:** Finding the 'Goldilocks' model complexity."},
            {"question": "Purpose of Cross-Validation.", "answer": "1. **Definition:** A technique to assess how the model will generalize to an independent dataset.\n2. **Explanation:** Rotates training and testing data to ensure every point is used for both. Gives more robust accuracy.\n3. **Example:** 10-Fold CV is the industry standard."},
            {"question": "MSE vs MAE.", "answer": "1. **Definition:** Metrics for Regression errors.\n2. **Explanation:** MSE squashes small errors but punishes large errors heavily (Square). MAE treats all errors linearly (Absolute).\n3. **Example:** Use MAE if you have outliers you don't want to over-penalize."},
            {"question": "Describe the 'Kernel Trick'.", "answer": "1. **Definition:** A method to solve non-linear problems with a linear classifier.\n2. **Explanation:** Maps low-dimensional input space to high-dimensional feature space where data becomes linearly separable.\n3. **Example:** RBF Kernel separating concentric circles of data."},
            {"question": "Steps to Build a Classifier.", "answer": "1. **Definition:** The workflow for supervised classification.\n2. **Explanation:** 1. Preprocess Data. 2. Split (Train/Test). 3. Init Model (e.g., SVC). 4. Fit(Train). 5. Predict(Test).\n3. **Example:** Building a Spam Filter using Scikit-Learn steps."},
            {"question": "Gradient Descent intuition.", "answer": "1. **Definition:** An iterative optimization algorithm.\n2. **Explanation:** Imagine walking down a mountain blindfolded. You take steps in the direction of the steepest slope downwards.\n3. **Example:** Used to update weights in Neural Networks."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Linear Regression.",
                "answer": "1. **Definition:**\n   The simplest form of supervised learning used to predict a continuous dependent variable based on independent variables.\n\n2. **Goal:**\n   To find the best-fitting straight line (Regression Line) that minimizes the error between actual and predicted values.\n\n3. **Core Concept:**\n   Equation: Y = mX + c (Simple) or Y = b0 + b1X1 + ... (Multiple). Uses 'Least Squares' method.\n\n4. **Technique / Method:**\n   Calculate coefficients weights (m) and bias (c) to minimize the Sum of Squared Errors (SSE).\n\n5. **Applications:**\n   Predicting sales based on ad spend, Estimating house prices, Forecasting temperature."
            },
            {
                "question": "Detailed Explanation of Logistic Regression.",
                "answer": "1. **Definition:**\n   A statistical model used for binary classification tasks (predicting one of two possible outcomes).\n\n2. **Goal:**\n   To estimate the probability that an instance belongs to a specific class (0 or 1).\n\n3. **Core Concept:**\n   Uses the Sigmoid (Logistic) Function to transform any input into a value between 0 and 1.\n\n4. **Technique / Method:**\n   If Output > 0.5, Class 1. If Output < 0.5, Class 0. Uses 'Log Loss' (Cross-Entropy) for optimization.\n\n5. **Applications:**\n   Spam detection (Spam/Not Spam), Credit Default prediction, Disease diagnosis."
            },
            {
                "question": "Detailed Explanation of Support Vector Machines (SVM).",
                "answer": "1. **Definition:**\n   A powerful supervised learning algorithm used for both classification and regression.\n\n2. **Goal:**\n   To find a hyperplane in an N-dimensional space that distinctly classifies the data points.\n\n3. **Core Concept:**\n   Margin Maximization. It tries to maximize the distance between the hyperplane and the nearest data points (Support Vectors).\n\n4. **Technique / Method:**\n   Uses Kernels (Linear, Polynomial, RBF) to handle non-linear data by projecting it to higher dimensions.\n\n5. **Applications:**\n   Text Categorization, Image Classification, Handwriting Recognition."
            },
            {
                "question": "Detailed Explanation of Decision Trees.",
                "answer": "1. **Definition:**\n   A non-parametric model that predicts the value of a target variable by learning simple decision rules.\n\n2. **Goal:**\n   To create a model that predicts the class or value of a target variable by learning simple decision rules inferred from the data features.\n\n3. **Core Concept:**\n   Tree structure: Root Node -> Internal Nodes (Splits) -> Leaf Nodes (Final Output). Recursive Partitioning.\n\n4. **Technique / Method:**\n   Splits are chosen to maximize Information Gain (Entropy reduction) or Gini Impurity reduction.\n\n5. **Applications:**\n   Loan Approval prediction, Customer Churn analysis, Medical decision support."
            },
            {
                "question": "Detailed Explanation of Random Forest.",
                "answer": "1. **Definition:**\n   An ensemble learning method that constructs a multitude of decision trees at training time.\n\n2. **Goal:**\n   To correct for the habit of decision trees to overfit to their training set.\n\n3. **Core Concept:**\n   'Wisdom of Crowds'. Many weak learners (trees) come together to make a strong learner.\n\n4. **Technique / Method:**\n   Bagging (Bootstrap Aggregating): Random samples of data + Random subsets of features for each tree.\n\n5. **Applications:**\n   Stock market behavior analysis, Banking fraud detection, E-commerce recommendation engines."
            },
            {
                "question": "Detailed Explanation of Naïve Bayes Classifier.",
                "answer": "1. **Definition:**\n   A probabilistic machine learning model based on applying Bayes' Theorem.\n\n2. **Goal:**\n   To classify data points based on the probability of a hypothesis being true given the evidence.\n\n3. **Core Concept:**\n   'Naïve' because it assumes strong independence between features (e.g., 'Red' is unrelated to 'Round').\n\n4. **Technique / Method:**\n   Calculates P(Class|Data) = [P(Data|Class) * P(Class)] / P(Data). High scalability.\n\n5. **Applications:**\n   Sentiment Analysis, Spam Filtering, Document Categorization."
            },
            {
                "question": "Detailed Explanation of K-Nearest Neighbors (KNN).",
                "answer": "1. **Definition:**\n   A simple, non-parametric, lazy learning algorithm used for classification and regression.\n\n2. **Goal:**\n   To assume that similar things exist in close proximity and classify new data based on neighbors.\n\n3. **Core Concept:**\n   Birds of a feather flock together. The class is determined by the majority vote of the nearest neighbors.\n\n4. **Technique / Method:**\n   1. Choose K. 2. Calculate distance (Euclidean). 3. Find K nearest. 4. Vote. 5. Assign Class.\n\n5. **Applications:**\n   Recommender Systems (Movies/Products), Pattern Recognition, Gene Expression analysis."
            },
            {
                "question": "Detailed Explanation of Evaluation Metrics (Classification).",
                "answer": "1. **Definition:**\n   Standard measures used to assess the quality and performance of a classification model.\n\n2. **Goal:**\n   To quantify how well the model performs on unseen data beyond just simple 'Accuracy'.\n\n3. **Core Concept:**\n   Different errors have different costs (e.g., missing a cancer diagnosis is worse than a false alarm).\n\n4. **Technique / Method:**\n   Key metrics: Precision (Quality), Recall (Quantity), F1-Score (Balance), and Confusion Matrix.\n\n5. **Applications:**\n   Used in every supervised learning project to validate model prior to deployment."
            },
            {
                "question": "Overfitting and Underfitting: Concepts and Solutions.",
                "answer": "1. **Definition:**\n   The two most common causes of poor model performance related to model complexity.\n\n2. **Goal:**\n   To achieve a 'Good Fit' that generalizes well to new, unseen data.\n\n3. **Core Concept:**\n   Overfitting: High Variance (Memorizes noise). Underfitting: High Bias (Too simple/Misses patterns).\n\n4. **Technique / Method:**\n   Fix Overfitting: More Data, Regularization, Pruning, Dropouts. Fix Underfitting: More features, Complex model.\n\n5. **Applications:**\n   Critical for diagnosing why a model fails during the testing/validation phase."
            },
            {
                "question": "Detailed Explanation of Gradient Descent.",
                "answer": "1. **Definition:**\n   A first-order iterative optimization algorithm used to find the minimum of a function.\n\n2. **Goal:**\n   To find the optimal parameters (weights) of a model that minimize the cost/loss function.\n\n3. **Core Concept:**\n   Iteratively moving in the direction of steepest descent (negative of the gradient).\n\n4. **Technique / Method:**\n   Update rule: New_Weight = Old_Weight - (Learning_Rate * Gradient). Rate controls step size.\n\n5. **Applications:**\n   The backbone of training Neural Networks, Linear Regression, and Logistic Regression."
            }
        ]
    }
}

def populate_ml_unit2():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Unit 2: Supervised Learning" to ensure clean slate
        data = [u for u in data if "Unit 2: Supervised Learning" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(ML_UNIT_2_DATA)
        print("Successfully replaced ML Unit 2 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_ml_unit2()
