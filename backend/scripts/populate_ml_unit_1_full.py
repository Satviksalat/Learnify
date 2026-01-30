
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR ML UNIT 1
ML_UNIT_1_DATA = {
    "unit": "ML Unit 1: Introduction to ML",
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
            {"question": "What is 'Noise' in data?", "answer": "Irrelevant or meaningless data that can confuse the model."},
            {"question": "Who coined the term 'Machine Learning'?", "answer": "Arthur Samuel (1959)."},
            {"question": "Define 'Model' in ML.", "answer": "The mathematical representation of a real-world process learned from data."},
            {"question": "Define 'Algorithm' in ML.", "answer": "A set of rules or calculations used to solve a problem or learn from data."},
            {"question": "What is Inductive Learning?", "answer": "Learning general rules from specific examples (Bottom-up approach)."},
            {"question": "What is Deductive Learning?", "answer": "Using general rules to draw specific conclusions (Top-down approach)."},
            {"question": "What represents 'E' in Mitchell's definition?", "answer": "Experience: The past data or interaction the system learns from."},
            {"question": "What represents 'T' in Mitchell's definition?", "answer": "Task: The specific goal or job the system is trying to accomplish."},
            {"question": "What represents 'P' in Mitchell's definition?", "answer": "Performance: The metric used to measure how well the system does the task."}
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
            {"question": "Role of an Algorithm vs Model.", "answer": "• Algorithm: The procedure (recipe) used to learn.\n• Model: The result (cooked dish) of training the algorithm on data."},
            {"question": "What is a Validation Set?", "answer": "• A subset of data used during training to tune hyperparameters.\n• Distinct from Test set (final eval) and Training set (learning)."},
            {"question": "Discrete vs Continuous Data.", "answer": "• Discrete: Distinct values (Integers, Categories).\n• Continuous: Infinite values within a range (Floats, Measurements)."},
            {"question": "Structured vs Unstructured Data.", "answer": "• Structured: Organized in tables/rows/cols (Excel, SQL).\n• Unstructured: No fixed format (Images, Text, Audio)."}
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
            {"question": "What happens during 'Prediction' phase?", "answer": "1. **Definition:** The final stage where the model is deployed for real use.\n2. **Explanation:** We feed it new, real-world live data, and it outputs the result based on training.\n3. **Example:** Netflix suggesting a movie you might like right now."},
            {"question": "What are Features of Good Data?", "answer": "1. **Definition:** Characteristics that make data suitable for training accurate models.\n2. **Explanation:** Data must be Relevant (related to problem), Accurate (trustworthy), and Complete (no missing parts).\n3. **Example:** Using current house prices, not 1990 prices, to predict 2024 values."},
            {"question": "Major Challenges in ML.", "answer": "1. **Definition:** Obstacles that hinder the performance or development of ML models.\n2. **Explanation:** 1. Insufficient Data. 2. Poor Quality Data (Noise). 3. Overfitting (Lack of generalization).\n3. **Example:** Training a self-driving car only in sunny weather fails in rain."},
            {"question": "Traditional Programming vs Machine Learning.", "answer": "1. **Definition:** Two different paradigms of solving problems with software.\n2. **Explanation:** Traditional: Input + Rules = Output. ML: Input + Output = Rules (Learning).\n3. **Example:** Trad: Hardcoding tax rules. ML: Learning tax patterns from history."},
            {"question": "Human vs Machine Learning.", "answer": "1. **Definition:** Comparing biological learning vs computational learning.\n2. **Explanation:** Humans learn from little data (1-2 examples) using intuition. Machines need massive data but process it faster.\n3. **Example:** A child sees 1 cat and knows cats. An ML model needs 1000 cat images."},
            {"question": "Importance of Feature Extraction.", "answer": "1. **Definition:** Process of transforming raw data into meaningful numerical features.\n2. **Explanation:** Raw data (pixels, audio) is too complex. We extract key traits (edges, frequencies) to simplify learning.\n3. **Example:** Extracting specific words (keywords) from an email to detect spam."},
            {"question": "Bias vs Variance in simple terms.", "answer": "1. **Definition:** Two sources of error in ML models.\n2. **Explanation:** Bias: Error from erroneous assumptions (Underfitting). Variance: Error from sensitivity to noise (Overfitting).\n3. **Example:** Bias is ignoring data. Variance is memorizing noise."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Explain the 7 Steps of Machine Learning.",
                "answer": "1. **Definition:**\n   The standard lifecycle to create an effective Machine Learning model.\n\n2. **Goal:**\n   To ensure a structured approach from Raw Data to a deployed, working prediction system.\n\n3. **Core Concept:**\n   Iterative process: Gather -> Clean -> Choose -> Train -> Test -> Tune -> Predict.\n\n4. **Technique / Method:**\n   Use tools like Pandas for cleaning, Scikit-Learn for training/testing, and GridSearch for tuning.\n\n5. **Applications:**\n   This workflow is used in every ML project, from simple student projects to enterprise AI."
            },
            {
                "question": "Discuss Real-World Applications of Machine Learning.",
                "answer": "1. **Definition:**\n   Practical implementations of ML algorithms to solve complex human problems.\n\n2. **Goal:**\n   To automate tasks, find hidden insights, and improve decision-making speed.\n\n3. **Core Concept:**\n   Training models on historical data to predict future outcomes.\n\n4. **Technique / Method:**\n   Examples: Recommendation Systems (Collaborative Filtering), Diagnostics (Image Classification).\n\n5. **Applications:**\n   Healthcare (Tumor detection), Finance (Fraud check), Transport (Self-driving), Marketing (Ads)."
            },
            {
                "question": "Comparison: Supervised vs Unsupervised Learning.",
                "answer": "1. **Definition:**\n   The two primary categories of Machine Learning based on the nature of input data.\n\n2. **Goal:**\n   Supervised learns specific answers. Unsupervised finds hidden structures.\n\n3. **Core Concept:**\n   Presence of Labels (Supervised) vs Absence of Labels (Unsupervised).\n\n4. **Technique / Method:**\n   Supervised: Regression, SVM (Teacher). Unsupervised: K-Means, PCA (Self-Learning).\n\n5. **Applications:**\n   Supervised: FaceID, Spam Filters. Unsupervised: Customer Grouping, Anomaly Detection."
            },
            {
                "question": "Explain the hierarchy of AI, ML, and DL with a diagram description.",
                "answer": "1. **Definition:**\n   A hierarchical relationship where Deep Learning is a subset of Machine Learning, which is a subset of Artificial Intelligence.\n\n2. **Goal:**\n   To classify intelligent systems based on their complexity, data requirements, and underlying techniques.\n\n3. **Core Concept:**\n   Visualized as concentric circles: AI (Outer / Broadest), ML (Middle / Learned), DL (Inner / Deep Neural Networks).\n\n4. **Technique / Method:**\n   AI uses rules/logic. ML uses statistical algorithms. DL uses Multi-layered Neural Networks.\n\n5. **Applications:**\n   AI: Symbolic Logic, Chess. ML: Spam Filters. DL: Generative AI (ChatGPT), Self-driving cars."
            },
            {
                "question": "Detailed Explanation of Reinforcement Learning.",
                "answer": "1. **Definition:**\n   A type of dynamic learning where an agent learns to make decisions by performing actions in an environment.\n\n2. **Goal:**\n   To maximize the total cumulative 'Reward' over time.\n\n3. **Core Concept:**\n   No predefined data. Learning happens via interaction loops: Observation -> Action -> Reward/Penalty -> Update Policy.\n\n4. **Technique / Method:**\n   Uses Q-Learning or Deep Q-Networks. The agent explores (tries new things) and exploits (uses known good actions).\n\n5. **Applications:**\n   Game playing (AlphaGo), Robotics (Walking robots), and Self-driving car navigation."
            },
            {
                "question": "Detailed Explanation of Deep Learning.",
                "answer": "1. **Definition:**\n   A specialized subset of Machine Learning that uses multi-layered artificial neural networks.\n\n2. **Goal:**\n   To mimic the human brain's ability to learn and classify from vast amounts of unstructured data.\n\n3. **Core Concept:**\n   Automatic Feature Extraction. Earlier layers find edges, middle layers find shapes, deep layers find objects.\n\n4. **Technique / Method:**\n   Uses Backpropagation to adjust weights in networks like CNNs (Images) or RNNs/Transformers (Text).\n\n5. **Applications:**\n   Complex tasks like Speech Recognition, Image Generation, and Language Translation."
            },
            {
                "question": "Detailed Explanation of Semi-Supervised Learning.",
                "answer": "1. **Definition:**\n   A machine learning approach that combines a small amount of labeled data with a large amount of unlabeled data.\n\n2. **Goal:**\n   To improve learning accuracy when labeling data is too expensive or time-consuming.\n\n3. **Core Concept:**\n   Use small labeled data to train a basic model, then use that to predict labels for the unlabeled data (Pseudo-labeling).\n\n4. **Technique / Method:**\n   1. Train on labeled. 2. Predict on unlabeled with high confidence. 3. Add to training set. 4. Retrain.\n\n5. **Applications:**\n   Text Classification (few tagged documents), Medical Imaging (few expert diagnoses)."
            },
            {
                "question": "Why do we need Machine Learning? (Why now?)",
                "answer": "1. **Definition:**\n   The necessity of ML arises from the limitations of traditional programming in handling modern data scale.\n\n2. **Goal:**\n   To solve problems that are too complex for hard-coded rules and to interpret 'Big Data'.\n\n3. **Core Concept:**\n   Data Explosion + Computation Power. We now have the data to teach machines and the GPUs to process it.\n\n4. **Technique / Method:**\n   Replace manual rule-writing (if-else) with algorithms that auto-generate rules from petabytes of data.\n\n5. **Applications:**\n   Personalization (Netflix), Search Engines (Google), and complex Pattern Recognition."
            },
            {
                "question": "Challenges and Limitations of Machine Learning.",
                "answer": "1. **Definition:**\n   The technical, ethical, and practical hurdles faced when developing and deploying ML systems.\n\n2. **Goal:**\n   To identify risks like bias, data privacy, and resource costs to build responsible AI.\n\n3. **Core Concept:**\n   garbage In, Garbage Out. Models are only as good as the data they are fed.\n\n4. **Technique / Method:**\n   Issues include Overfitting, Data Bias (Gender/Race), Lack of Explainability (Black Box), and high GPU costs.\n\n5. **Applications:**\n   Critical systems like Hiring Tools or Autonomous Weapons where errors are unacceptable."
            },
            {
                "question": "Traditional Programming vs Machine Learning in Detail.",
                "answer": "1. **Definition:**\n   A comparative analysis of the classic software development approach versus the modern data-driven approach.\n\n2. **Goal:**\n   To understand when to use coding logic versus when to use statistical learning.\n\n3. **Core Concept:**\n   Paradigm Shift. Trad: Programmer gives rules. ML: Programmer gives data, Machine discerns rules.\n\n4. **Technique / Method:**\n   Trad: Logic-based, Deterministic, explicit. ML: Probability-based, Stochastic, implicit.\n\n5. **Applications:**\n   Use Trad for Banking Ledgers (Exact). Use ML for Credit Scoring (Probabilistic)."
            }
        ]
    }
}

def populate_ml_unit1():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Introduction to ML" unit (fix naming variants)
        # We filter out any unit containing "Introduction to ML" to ensure a clean slate
        data = [u for u in data if "Introduction to ML" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(ML_UNIT_1_DATA)
        print("Successfully replaced ML Unit 1 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_ml_unit1()
