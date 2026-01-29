import json
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

tutorials_path = os.path.join(DATA_DIR, 'tutorials.json')
quizzes_path = os.path.join(DATA_DIR, 'quizzes.json')
programs_path = os.path.join(DATA_DIR, 'programs.json')
exams_path = os.path.join(DATA_DIR, 'exam_questions.json')

# --- 1. NEW DEEP LEARNING TUTORIALS ---
dl_tutorials = [
    # UNIT 1: INTRO TO DEEP LEARNING (Replaces Intro to ML)
    {
        "id": "dl-unit1-intro",
        "title": "1. What is Deep Learning?",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Deep Learning Foundations",
        "definition": "Deep Learning is a subset of ML based on ANNs.",
        "description": "Understanding the hierarchy: AI > ML > DL.",
        "explanation": "<h2>1. The AI Hierarchy</h2><p><strong>Artificial Intelligence (AI):</strong> Machines mimicking human intelligence.</p><p><strong>Machine Learning (ML):</strong> Algorithms that learn from data.</p><p><strong>Deep Learning (DL):</strong> Neural networks with many layers dealing with vast unstructured data.</p><h2>2. Why Deep Learning?</h2><p>Traditional ML plateaus with more data, but DL performance keeps increasing. It excels in Image Recognition, NLP, and Speech.</p>",
        "key_points": ["Neural Networks", "Big Data", "GPU Acceleration"]
    },
    {
        "id": "dl-unit1-limitations",
        "title": "2. History & Limitations",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Deep Learning Foundations",
        "definition": "From Perceptrons to Transformers.",
        "description": "A brief history and current challenges.",
        "explanation": "<h2>1. A Brief History</h2><ul><li><strong>1958:</strong> Perceptron invented (Single layer).</li><li><strong>1986:</strong> Backpropagation popularized (Hinton).</li><li><strong>2012:</strong> AlexNet wins ImageNet (DL Boom).</li></ul><h2>2. Limitations</h2><p>Data Hungry, Computationally Expensive (needs GPUs), and 'Black Box' nature (hard to interpret).</p>",
        "key_points": ["Perceptron", "AlexNet", "Data Hungry"]
    },

    # UNIT 2: NEURAL NETWORKS (Replaces Supervised Learning)
    {
        "id": "dl-unit2-neuron",
        "title": "1. The Artificial Neuron",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Artificial Neural Networks",
        "definition": "Mathematical model of a biological neuron.",
        "description": "Inputs, Weights, Bias, and Activation.",
        "explanation": "<h2>1. Structure of a Neuron</h2><p>$$ z = \\sum (x_i \\cdot w_i) + b $$</p><p>Where <strong>x</strong> are inputs, <strong>w</strong> are weights, and <strong>b</strong> is bias. The output is passed through an activation function.</p><h2>2. Activation Functions</h2><p>Decides if a neuron should 'fire'. Common types: Sigmoid, ReLU, Tanh.</p>",
        "key_points": ["Weights", "Bias", "Dot Product"]
    },
    {
        "id": "dl-unit2-mlp",
        "title": "2. Multi-Layer Perceptron (MLP)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Artificial Neural Networks",
        "definition": "Fully Connected Networks.",
        "description": "Layers: Input, Hidden, and Output.",
        "explanation": "<h2>1. Architecture</h2><p>Stacked layers of neurons. 'Deep' means multiple hidden layers.</p><h2>2. Forward Propagation</h2><p>Data flows from Input -> Hidden -> Output layer to generate a prediction.</p><h2>3. Universal Approximation</h2><p>MLPs can theoretically learn any function given enough neurons.</p>",
        "key_points": ["Hidden Layers", "Forward Prop", "Dense Layers"]
    },
    {
        "id": "dl-unit2-training",
        "title": "3. Loss & Backpropagation",
        "technology": "Machine Learning with Python",
        "unit": "Unit 2: Artificial Neural Networks",
        "definition": "How networks learn.",
        "description": "Minimizing Error using Gradient Descent.",
        "explanation": "<h2>1. Loss Function</h2><p>Measures how wrong the prediction is (e.g., MSE, Cross-Entropy).</p><h2>2. Backpropagation</h2><p>The core algorithm. It calculates the gradient of the loss with respect to each weight using the Chain Rule.</p><h2>3. Optimizer</h2><p>Updates weights (e.g., SGD, Adam) to reduce loss.</p>",
        "key_points": ["Gradient Descent", "Chain Rule", "Learning Rate"]
    },

    # UNIT 3: CNN (Replaces Unsupervised Learning)
    {
        "id": "dl-unit3-cnn-intro",
        "title": "1. Intro to CNNs",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Convolutional Neural Networks",
        "definition": "Specialized for Image Data.",
        "description": "Why MLPs fail on images.",
        "explanation": "<h2>1. The Problem with MLPs</h2><p>Flattening images destroys spatial structure and creates too many parameters.</p><h2>2. The Solution: Convolutions</h2><p>Sliding a filter (kernel) over the image to detect features like edges, textures, and shapes while preserving spatial relationships.</p>",
        "key_points": ["Spatial Invariance", "Filters", "Kernels"]
    },
    {
        "id": "dl-unit3-layers",
        "title": "2. CNN Layers",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Convolutional Neural Networks",
        "definition": "Conv, Pool, FC.",
        "description": "Building blocks of a CNN.",
        "explanation": "<h2>1. Convolutional Layer</h2><p>Extracts features using filters.</p><h2>2. Pooling Layer</h2><p>Reduces dimensionality/size (e.g., Max Pooling) to reduce computation.</p><h2>3. Fully Connected (FC) Layer</h2><p>The final classification layer (like an MLP) at the end of the network.</p>",
        "key_points": ["Feature Map", "Max Pooling", "Stride"]
    },
    {
        "id": "dl-unit3-transfer",
        "title": "3. Transfer Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 3: Convolutional Neural Networks",
        "definition": "Using Pre-trained Models.",
        "description": "VGG, ResNet, Inception.",
        "explanation": "<h2>1. Concept</h2><p>Don't train from scratch! Use models trained on ImageNet.</p><h2>2. Fine Tuning</h2><p>Freeze the early layers (which detect edges) and only train the final layers for your specific dataset.</p>",
        "key_points": ["Pre-trained", "Frozen Layers", "ImageNet"]
    },

    # UNIT 4: RNN (Replaces NLP)
    {
        "id": "dl-unit4-rnn-intro",
        "title": "1. Intro to RNNs",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Recurrent Neural Networks",
        "definition": "For Sequential Data.",
        "description": "Handling Time-Series and Text.",
        "explanation": "<h2>1. Sequential Data</h2><p>Text, Audio, Stock Prices. Order matters.</p><h2>2. RNN Architecture</h2><p>Has a 'memory' loop. The output of the previous step is fed as input to the next step.</p><h2>3. Vanishing Gradient</h2><p>The main problem with basic RNNs\u2014they forget long-term dependencies.</p>",
        "key_points": ["Sequence", "Hidden State", "Time Steps"]
    },
    {
        "id": "dl-unit4-lstm",
        "title": "2. LSTMs & GRUs",
        "technology": "Machine Learning with Python",
        "unit": "Unit 4: Recurrent Neural Networks",
        "definition": "Long Short-Term Memory.",
        "description": "Advanced RNNs.",
        "explanation": "<h2>1. LSTM Cell</h2><p>Uses 'Gates' (Forget, Input, Output) to control information flow. Solves the Vanishing Gradient problem.</p><h2>2. GRU</h2><p>Gated Recurrent Unit. A simplified version of LSTM with fewer parameters.</p>",
        "key_points": ["Gates", "Memory Cell", "Long-term Dependency"]
    },

    # UNIT 5: ADVANCED DL (Replaces CV)
    {
        "id": "dl-unit5-transformers",
        "title": "1. Transformers & Attention",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Advanced Deep Learning",
        "definition": "State of the Art NLP.",
        "description": "BERT, GPT, and Self-Attention.",
        "explanation": "<h2>1. Need for Attention</h2><p>RNNs are slow (sequential). Transformers process the whole sequence at once (Parallel).</p><h2>2. Self-Attention</h2><p>Mechanism to weigh the importance of different words in a sentence relative to each other.</p>",
        "key_points": ["Parallelization", "Keys/Queries/Values", "BERT"]
    },
    {
        "id": "dl-unit5-gan",
        "title": "2. GANs (Generative AI)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 5: Advanced Deep Learning",
        "definition": "Generative Adversarial Networks.",
        "description": "Generator vs Discriminator.",
        "explanation": "<h2>1. The Game</h2><p>Two networks compete.</p><ul><li><strong>Generator:</strong> Creates fake images.</li><li><strong>Discriminator:</strong> Tries to distinguish fake from real.</li></ul><h2>2. Nash Equilibrium</h2><p>They improve each other until the fakes are indistinguishable from real data.</p>",
        "key_points": ["Generator", "Discriminator", "Fake Data"]
    }
]

# --- 2. NEW DEEP LEARNING QUIZZES ---
dl_quizzes = [
    {"id": 101, "course": "Machine Learning", "unit": "Unit 1: Deep Learning Foundations", "question": "What is the primary difference between DL and Classical ML?", "options": ["DL uses Neural Networks", "DL uses Random Forests", "ML is faster", "There is no difference"], "correct_answer": "DL uses Neural Networks"},
    {"id": 102, "course": "Machine Learning", "unit": "Unit 1: Deep Learning Foundations", "question": "Who popularized Backpropagation?", "options": ["Geoffrey Hinton", "Alan Turing", "Elon Musk", "Andrew Ng"], "correct_answer": "Geoffrey Hinton"},
    
    {"id": 103, "course": "Machine Learning", "unit": "Unit 2: Artificial Neural Networks", "question": "What is the role of an Activation Function?", "options": ["To introduce non-linearity", "To speed up training", "To initialize weights", "To calculate loss"], "correct_answer": "To introduce non-linearity"},
    {"id": 104, "course": "Machine Learning", "unit": "Unit 2: Artificial Neural Networks", "question": "Which algorithm updates weights to minimize loss?", "options": ["Gradient Descent", "K-Means", "Random Search", "Decision Tree"], "correct_answer": "Gradient Descent"},
    
    {"id": 105, "course": "Machine Learning", "unit": "Unit 3: Convolutional Neural Networks", "question": "What does a pooling layer do?", "options": ["Reduces dimensionality", "Increases dimensionality", "Changes colors", "Inverts image"], "correct_answer": "Reduces dimensionality"},
    {"id": 106, "course": "Machine Learning", "unit": "Unit 3: Convolutional Neural Networks", "question": "What is ImageNet?", "options": ["A large image database", "A network architecture", "A python library", "A computer"], "correct_answer": "A large image database"},
    
    {"id": 107, "course": "Machine Learning", "unit": "Unit 4: Recurrent Neural Networks", "question": "What problem do LSTMs solve?", "options": ["Vanishing Gradient", "Overfitting", "Underfitting", "High Bias"], "correct_answer": "Vanishing Gradient"},
    {"id": 108, "course": "Machine Learning", "unit": "Unit 4: Recurrent Neural Networks", "question": "Which data type is best for RNNs?", "options": ["Sequential/Time-series", "Images", "Tabular", "Graphs"], "correct_answer": "Sequential/Time-series"},
    
    {"id": 109, "course": "Machine Learning", "unit": "Unit 5: Advanced Deep Learning", "question": "What mechanism do Transformers use?", "options": ["Self-Attention", "Convolutions", "Recurrence", "Randomness"], "correct_answer": "Self-Attention"},
    {"id": 110, "course": "Machine Learning", "unit": "Unit 5: Advanced Deep Learning", "question": "In GANs, what does the Discriminator do?", "options": ["Classifies Real vs Fake", "Generates Fake Data", "Optimizes Loss", "Preprocesses Data"], "correct_answer": "Classifies Real vs Fake"}
]

# --- 3. NEW DEEP LEARNING EXAMS ---
dl_exams = [
    {
        "unit": "Unit 1: Deep Learning Foundations",
        "sections": {
            "Part A": [
                {"question": "Define Deep Learning.", "answer": "A subset of ML based on ANNs with multiple layers."},
                {"question": "What is a Perceptron?", "answer": "The simplest type of artificial neuron."}
            ],
            "Part B": [
                {"question": "Distinguish between Shallow and Deep Neural Networks.", "answer": "Shallow has 0-1 hidden layers, Deep has many."}
            ]
        }
    },
    {
        "unit": "Unit 2: Artificial Neural Networks",
        "sections": {
            "Part A": [
                {"question": "What is ReLU?", "answer": "Rectified Linear Unit. Activation function f(x)=max(0,x)."},
                {"question": "What is a bias?", "answer": "An extra parameter allowing the activation to shift."},
                {"question": "What is an Epoch?", "answer": "One complete pass of the training dataset."}
            ],
            "Part B": [
                {"question": "Explain Gradient Descent.", "answer": "Optimization algorithm to find the minimum of a function (Loss). Steps downhill."},
                {"question": "Why do we need Non-Linearity?", "answer": "Without it, a Neural Network is just a Linear Regression model."}
            ]
        }
    },
    {
        "unit": "Unit 3: Convolutional Neural Networks",
        "sections": {
            "Part A": [
                {"question": "What is a Kernel?", "answer": "A filter used to extract features from an image."},
                {"question": "What is Stride?", "answer": "The number of pixels the filter moves at each step."}
            ],
            "Part B": [
                {"question": "Explain Max Pooling.", "answer": "Taking the maximum value from a feature map region. Reduces size and params."},
                {"question": "What is Padding?", "answer": "Adding zeros around the input to preserve output dimensions."}
            ]
        }
    },
    {
        "unit": "Unit 4: Recurrent Neural Networks",
        "sections": {
            "Part A": [
                {"question": "What is a Hidden State?", "answer": "The memory of the network from the previous time step."},
                {"question": "What is Backpropagation Through Time (BPTT)?", "answer": "Standard backprop applied to unfolded RNNs."}
            ],
            "Part B": [
                {"question": "Difference between LSTM and RNN.", "answer": "LSTM has gates to control memory flow, standard RNN does not."}
            ]
        }
    },
    {
        "unit": "Unit 5: Advanced Deep Learning",
        "sections": {
            "Part A": [
                {"question": "What is BERT?", "answer": "Bidirectional Encoder Representations from Transformers."},
                {"question": "What is a Generator in GAN?", "answer": "The network that creates fake data to fool the discriminator."}
            ],
            "Part B": [
                {"question": "Explain 'Attention' mechanism.", "answer": "Allows the model to focus on relevant parts of the input sequence regardless of distance."}
            ]
        }
    }
]

# --- 4. NEW DEEP LEARNING PROGRAMS ---
dl_programs = [
    {
        "id": 201, "course": "Machine Learning", "unit": "Unit 2: Artificial Neural Networks",
        "question": "Implement a simple neuron in Python.",
        "code": "import numpy as np\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n\ninputs = np.array([0.5, 0.1])\nweights = np.array([0.4, 0.8])\nbias = 0.5\n\noutput = sigmoid(np.dot(inputs, weights) + bias)\nprint(output)"
    },
    {
        "id": 202, "course": "Machine Learning", "unit": "Unit 3: Convolutional Neural Networks",
        "question": "Create a basic CNN model structure using Keras.",
        "code": "from tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense\n\nmodel = Sequential([\n    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),\n    MaxPooling2D(2,2),\n    Flatten(),\n    Dense(128, activation='relu'),\n    Dense(1, activation='sigmoid')\n])\nmodel.summary()"
    }
]

# --- EXECUTION FUNCTIONS ---

def update_tutorials():
    print("Updating Tutorials...")
    with open(tutorials_path, 'r') as f:
        data = json.load(f)
    
    # Filter out OLD Machine Learning content (anything starting with ml-unit or course="Machine Learning")
    # Actually, let's keep Python content and remove old ML content
    new_data = [t for t in data if t.get('technology') != "Machine Learning with Python"]
    
    # Append NEW DL content
    new_data.extend(dl_tutorials)
    
    with open(tutorials_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print("Tutorials Updated.")

def update_quizzes():
    print("Updating Quizzes...")
    with open(quizzes_path, 'r') as f:
        data = json.load(f)
    
    # Remove old ML quizzes
    new_data = [q for q in data if q.get('course') != "Machine Learning"]
    
    # Append NEW DL quizzes
    new_data.extend(dl_quizzes)
    
    with open(quizzes_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print("Quizzes Updated.")

def update_exams():
    print("Updating Exams...")
    with open(exams_path, 'r') as f:
        data = json.load(f)

    # Note: data in exams.json is a list of objects with "unit" keys.
    # We want to remove objects where 'unit' starts with "Unit X: ..." where X matches old ML units
    # Or simpler: Remove existing ML units and append new ones.
    
    # We'll filter based on the presence of Keywords or just assume last 5 are ML.
    # Safest: Filter out specific Old Unit Names.
    old_ml_units = [
        "Unit 1: Introduction to ML",
        "Unit 2: Supervised Learning", 
        "Unit 3: Unsupervised Learning",
        "Unit 4: Natural Language Processing",
        "Unit 5: Computer Vision with OpenCV",
        "Unit 4: NLP"
    ]
    
    new_data = [e for e in data if e.get('unit') not in old_ml_units]
    
    # Append NEW DL Exams
    new_data.extend(dl_exams)
    
    with open(exams_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print("Exams Updated.")

def update_programs():
    print("Updating Programs...")
    with open(programs_path, 'r') as f:
        data = json.load(f)
    
    # Remove old ML programs
    new_data = [p for p in data if p.get('course') != "Machine Learning"]
    
    # Append NEW DL programs
    new_data.extend(dl_programs)
    
    with open(programs_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print("Programs Updated.")

if __name__ == "__main__":
    update_tutorials()
    update_quizzes()
    update_exams()
    update_programs()
