import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# EXPERT CONTENT FOR UNIT 1
UNIT_1_DATA = {
    "unit": "Unit 1: Introduction to ML",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Machine Learning.", "answer": "ML is a field of AI where computers learn from data without being explicitly programmed."},
            {"question": "What is Artificial Intelligence?", "answer": "AI is the simulation of human intelligence processes by machines, especially computer systems."},
            {"question": "Define Deep Learning.", "answer": "A subset of ML based on artificial neural networks with multiple layers (Deep Neural Networks)."},
            {"question": "What is a 'Label' in ML?", "answer": "The target variable or the answer that the model is trying to predict."},
            {"question": "Define 'Feature'.", "answer": "An individual measurable property or characteristic of a phenomenon being observed."},
            {"question": "What is a Training Set?", "answer": "The subset of the dataset used to train or fit the model."},
            {"question": "What is a Testing Set?", "answer": "The subset of the dataset used to evaluate the trained model's performance."},
            {"question": "Define Supervised Learning.", "answer": "A type of learning where the model is trained on labeled data (Input-Output pairs)."},
            {"question": "Define Unsupervised Learning.", "answer": "Learning where the model works on unlabeled data to find hidden patterns or structures."},
            {"question": "What is Reinforcement Learning?", "answer": "A learning method where an agent learns by interacting with an environment and receiving rewards."},
            {"question": "What is a Dataset?", "answer": "A collection of data samples used for training and testing ML models."},
            {"question": "List one example of Supervised Learning.", "answer": "Spam Email Filtering (classifying email as 'Spam' or 'Not Spam')."},
            {"question": "List one example of Unsupervised Learning.", "answer": "Customer Segmentation (Grouping customers based on purchasing behavior)."},
            {"question": "What is 'Noise' in data?", "answer": "Irrelevant or meaningless data that can confuse the model."},
            {"question": "Who coined the term 'Machine Learning'?", "answer": "Arthur Samuel (1959)."}
        ],
        "Part B (2-Marks)": [
            {"question": "Differentiate AI and ML.", "answer": "• AI: Broader concept of smart machines mimicking human behavior.\n• ML: Subset of AI where machines learn from data to improve performance."},
            {"question": "State the difference between Labeled and Unlabeled data.", "answer": "• Labeled: Data comes with the correct answer/tag (e.g., Image of Cat + Label 'Cat').\n• Unlabeled: Raw data with no tags (e.g., Image of Cat with no info)."},
            {"question": "List 4 applications of Machine Learning.", "answer": "• Self-Driving Cars\n• Fraud Detection\n• Product Recommendation\n• Voice Assistants (Siri/Alexa)"},
            {"question": "What are the 3 main types of Learning?", "answer": "• Supervised Learning\n• Unsupervised Learning\n• Reinforcement Learning"},
            {"question": "Define 'Preprocessing' in ML.", "answer": "• Process of cleaning and transforming raw data.\n• Includes handling missing values, scaling, and encoding."},
            {"question": "What is the goal of the 'Training' phase?", "answer": "• To allow the model to learn patterns from the training data.\n• To minimize the error between predicted and actual results."},
            {"question": "Why do we split data into Train and Test?", "answer": "• To prevent Overfitting (memorizing data).\n• To evaluate how well the model acts on unseen (new) data."},
            {"question": "What is Sem-Supervised Learning?", "answer": "• Uses a small amount of labeled data and a large amount of unlabeled data.\n• Combines benefits of both Supervised and Unsupervised learning."},
            {"question": "Give 2 examples of Regression problems.", "answer": "• Predicting House Prices.\n• Predicting Stock Market trends."},
            {"question": "Give 2 examples of Classification problems.", "answer": "• Identifying tumors as Benign/Malignant.\n• Classifying images as Cat/Dog."},
            {"question": "What is 'Generalization'?", "answer": "• The ability of a model to adapt to new, unseen data.\n• A model with good generalization performs well on test data."},
            {"question": "What is the role of an 'Algorithm' in ML?", "answer": "• It is the mathematical logic/procedure used to find patterns in data.\n• Examples: Linear Regression, Decision Trees."}
        ],
        "Part C (3-Marks)": [
            {"question": "Explain Supervised Learning with an example.", "answer": "1. **Definition:** Learning where the machine is taught using data that is 'labeled'. The model maps Input (X) to Output (Y).\n2. **Explanation:** Like a teacher–student scenario. The teacher provides questions and answers. The student learns the logic.\n3. **Example:** Face Recognition (Input: Photo, Label: Name of person)."},
            {"question": "Explain Unsupervised Learning with an example.", "answer": "1. **Definition:** Learning where the machine is given data with NO labels. It must find structure on its own.\n2. **Explanation:** Like a baby learning about objects without a teacher. It groups similar things together.\n3. **Example:** News Categorization (Grouping similar articles by topic)."},
            {"question": "Explain Reinforcement Learning with an example.", "answer": "1. **Definition:** Learning by interacting with an environment to maximize rewards/minimize penalties.\n2. **Explanation:** Trial and error method. Good actions get +Points, bad actions get -Points.\n3. **Example:** Training a robot to walk (Falls = Penalty, Steps = Reward)."},
            {"question": "Comparing AI, ML, and DL.", "answer": "1. **AI:** The big umbrella. Any machine acting smart.\n2. **ML:** A subset of AI. Machines that simple learn from data (Statistical).\n3. **DL:** A subset of ML. Machines that learn via Neural Networks (Brain-like structure)."},
            {"question": "Explain 'Data Gathering' step in ML.", "answer": "1. **Definition:** The first and most crucial step of collecting raw data.\n2. **Explanation:** Quality and quantity of data determine model success. Identify sources (DB, API, Sensors).\n3. **Example:** Scraping tweets to analyze public sentiment."},
            {"question": "Explain 'Model Evaluation' step.", "answer": "1. **Definition:** Testing the trained model on a separate 'Test Set'.\n2. **Explanation:** Checking metrics like Accuracy. If performance is poor, we retune parameters.\n3. **Example:** Checking if a Spam filter catches 99% of spam emails."},
            {"question": "What is Deep Learning effectively?", "answer": "1. **Definition:** ML technique inspired by the human brain's neural networks.\n2. **Explanation:** Uses layers of algorithms to learn features from vast amounts of data. Good for unstructured data.\n3. **Example:** ChatGPT (Text generation), Tesla Autopilot (Vision)."},
            {"question": "Difference between Regression and Classification.", "answer": "1. **Regression:** Prediction of continuous/numerical value (e.g., Temperature 34.5°C).\n2. **Classification:** Prediction of categorical/discrete label (e.g., Hot/Cold).\n3. **Connection:** Both are types of Supervised Learning."},
            {"question": "What happens during 'Prediction' phase?", "answer": "1. **Definition:** The final stage where the model is deployed for real use.\n2. **Explanation:** We feed it new, real-world live data, and it outputs the result based on training.\n3. **Example:** Netflix suggesting a movie you might like right now."}
        ],
        "Part D (5-Marks)": [
            {"question": "Explain the 7 Steps of Machine Learning in detail.", "answer": "1. **Data Gathering:** Collecting information from various sources (Files, Databases, APIs).\n2. **Data Preprocessing:** Cleaning data (removing nulls, noise) and formatting it for the machine.\n3. **Choose Model:** Selecting the right algorithm (e.g., Regression for numbers, Decision Tree for rules).\n4. **Train Model:** The learning phase. The model processes data to find patterns.\n5. **Test Model:** Evaluating performance on unseen data to ensure it works.\n6. **Parameter Tuning:** Tweaking settings (Hyperparameters) to improve accuracy.\n7. **Prediction:** Deploying the model to solve the real-world problem."},
            {"question": "Discuss Real-World Applications of Machine Learning.", "answer": "1. **Healthcare:** Diagnosing diseases from X-Rays faster than doctors (Cancer detection).\n2. **Finance:** Fraud detection systems spotting unusual credit card transactions instantly.\n3. **E-commerce:** Recommendation engines (Amazon/Netflix) suggesting products based on history.\n4. **Social Media:** Face tagging in photos and content moderation (filtering hate speech).\n5. **Transport:** Self-driving cars (Tesla, Waymo) detecting lanes, cars, and pedestrians.\n6. **Why it matters:** Automates repetitive tasks and solves complex problems beyond human scale."},
            {"question": "Detailed comparison of Supervised vs Unsupervised Learning.", "answer": "1. **Labeling:** Supervised uses Labeled data (Input+Output). Unsupervised uses Unlabeled data (Input only).\n2. **Goal:** Supervised allows prediction/classification. Unsupervised allows discovery/clustering.\n3. **Complexity:** Supervised is generally easier to evaluate (compare to ground truth). Unsupervised is harder to evaluate (subjective).\n4. **Algorithms:** \n   - Supervised: Linear Regression, SVM, Decision Trees.\n   - Unsupervised: K-Means, Hierarchical Clustering, PCA.\n5. **Example:** \n   - Supervised: Teaching a child to name colors.\n   - Unsupervised: A child sorting blocks by shape without knowing the names."},
            {"question": "Explain the hierarchy of AI, ML, and DL with a diagram description.", "answer": "1. **Artificial Intelligence (AI):** The outermost circle. Concept of creating smart intelligent machines. (Birth: 1950s).\n2. **Machine Learning (ML):** Determining patterns from data. A subset inside AI. Uses statistical methods. (Boom: 1980s-90s).\n3. **Deep Learning (DL):** The innermost circle. Uses multi-layered Neural Networks. Needs huge data and compute power. (Boom: 2010s).\n\n**Visual:** An egg-like structure. Yolk is DL, White is ML, Shell is AI.\n**Key Takeaway:** All DL is ML, but not all ML is DL."}
        ]
    }
}

def update_unit1():
    try:
        # Load existing data
        if os.path.exists(EXAM_FILE):
             with open(EXAM_FILE, 'r') as f:
                data = json.load(f)
        else:
            print("Error: exam_questions.json not found.")
            return

        # Find and Update Unit 1 (ML)
        # We need to distinguish "Unit 1: Introduction to ML" from "Unit 1: Python Basics"
        updated = False
        for unit in data:
            if unit['unit'] == "Unit 1: Introduction to ML":
                unit['sections'] = UNIT_1_DATA['sections'] # Replace content
                print("Updated Unit 1: Introduction to ML with EXPERT content.")
                updated = True
                break
        
        if not updated:
            print("Unit 1: Introduction to ML not found in file. Appending...")
            data.append(UNIT_1_DATA)

        # Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Success.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_unit1()
