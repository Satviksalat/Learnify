import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

CONTENT_MAP = {
    # ---------------- UNIT 1: PYTHON BASICS ----------------
    "unit1-intro": """<h2>1. Concept Overview</h2><p><strong>Python</strong> is a high-level, interpreted, general-purpose programming language.</p><h2>2. Real-World Analogy</h2><p>Think of Python as <strong>English for Computers</strong>.</p>""",
    "unit1-syntax": """<h2>1. Concept Overview</h2><p><strong>Syntax</strong> refers to the grammatical rules. Python uses <strong>Indentation</strong>.</p>""",
    "unit1-vars": """<h2>1. Concept</h2><p>Variables are labels for data.</p><h2>2. Analogy</h2><p>Sticky Notes on objects.</p>""",
    "unit1-strings": """<h2>1. Concept</h2><p>Immutable sequences of characters.</p>""",
    "unit1-operators": """<h2>1. Concept</h2><p>Symbols for logic/math.</p>""",
    "unit1-ifelse": """<h2>1. Concept</h2><p>Branching logic.</p>""",
    "unit1-loops": """<h2>1. Concept</h2><p>Repetition.</p>""",
    "unit1-functions": """<h2>1. Concept</h2><p>Reusable blocks.</p>""",
    "unit1-lists-tuples": """<h2>1. Concept</h2><p>Sequences.</p>""",
    "unit1-dictionaries": """<h2>1. Concept</h2><p>Key-Value pairs.</p>""",
    "unit1-files": """<h2>1. Concept</h2><p>Persistence.</p>""",

    # ---------------- UNIT 2: OOP ----------------
    "unit2-oop-intro": """<h2>1. Concept</h2><p>OOP structure.</p>""",
    "unit2-classes": """<h2>1. Concept</h2><p>The Blueprint.</p>""",
    "unit2-constructors": """<h2>1. Concept</h2><p>Initializer `__init__`.</p>""",
    "unit2-inheritance": """<h2>1. Concept</h2><p>Parent-Child.</p>""",
    "unit2-encapsulation": """<h2>1. Concept</h2><p>Hiding data.</p>""",
    "unit2-polymorphism": """<h2>1. Concept</h2><p>Many forms.</p>""",
    "unit2-exceptions": """<h2>1. Concept</h2><p>Error handling.</p>""",
    "unit2-assertions": """<h2>1. Concept</h2><p>Debugging.</p>""",
    "unit2-search": """<h2>1. Concept</h2><p>Finding elements.</p>""",
    "unit2-sort": """<h2>1. Concept</h2><p>Ordering elements.</p>""",
    "unit2-hashing": """<h2>1. Concept</h2><p>Mapping keys.</p>""",

    # ---------------- UNIT 3: PLOTTING ----------------
    "unit3-pylab": """<h2>1. Concept</h2><p>PyLab module.</p>""",
    "unit3-lineplot": """<h2>1. Concept</h2><p>Line Plots.</p>""",
    "unit3-barchart": """<h2>1. Concept</h2><p>Bar Charts.</p>""",
    "unit3-mortgage": """<h2>1. Concept</h2><p>Amortization.</p>""",
    "unit3-fibonacci": """<h2>1. Concept</h2><p>Recursion demo.</p>""",
    "unit3-dp": """<h2>1. Concept</h2><p>Dynamic Programming.</p>""",
    "unit3-knapsack": """<h2>1. Concept</h2><p>Optimization.</p>""",
    "unit3-divideconquer": """<h2>1. Concept</h2><p>Divide and Conquer.</p>""",

    # ---------------- UNIT 4: NETWORKS ----------------
    "unit4-networkbasics": """<h2>1. Concept</h2><p>IP/Ports.</p>""",
    "unit4-sockets": """<h2>1. Concept</h2><p>Endpoints.</p>""",
    "unit4-tcp": """<h2>1. Concept</h2><p>Reliable stream.</p>""",
    "unit4-udp": """<h2>1. Concept</h2><p>Unreliable datagram.</p>""",
    "unit4-filetransfer": """<h2>1. Concept</h2><p>Chunked sending.</p>""",
    "unit4-email": """<h2>1. Concept</h2><p>SMTP.</p>""",
    "unit4-guiintro": """<h2>1. Concept</h2><p>Tkinter.</p>""",
    "unit4-buttons": """<h2>1. Concept</h2><p>Actions.</p>""",
    "unit4-entry": """<h2>1. Concept</h2><p>Inputs.</p>""",
    "unit4-layouts": """<h2>1. Concept</h2><p>Geometry Managers.</p>""",
    "unit4-treeview": """<h2>1. Concept</h2><p>Hierarchies.</p>""",

    # ---------------- UNIT 5: DATABASE ----------------
    "unit5-mysqlsetup": """<h2>1. Concept</h2><p>Driver install.</p>""",
    "unit5-connect": """<h2>1. Concept</h2><p>Connection object.</p>""",
    "unit5-create": """<h2>1. Concept</h2><p>DDL.</p>""",
    "unit5-insert": """<h2>1. Concept</h2><p>Insert data.</p>""",
    "unit5-select": """<h2>1. Concept</h2><p>Fetch data.</p>""",
    "unit5-update": """<h2>1. Concept</h2><p>Modify data.</p>""",
    "unit5-delete": """<h2>1. Concept</h2><p>Remove data.</p>""",

    # ================= ML UNIT 1: INTRO =================
    "ml-unit1-intro": """<h2>1. Concept Overview</h2><p>Machine Learning definitions.</p>""",
    "ml-unit1-ai-vs-ml": """<h2>1. Concept Overview</h2><p>AI > ML > DL.</p>""",
    "ml-unit1-howlearn": """<h2>1. Concept Overview</h2><p>Pattern Recognition.</p>""",
    "ml-unit1-steps": """<h2>1. Concept Overview</h2><p>7 Steps of ML.</p>""",
    "ml-unit1-types": """<h2>1. Concept Overview</h2><p>Sup/Unsup/RL.</p>""",
    "ml-unit1-supervised-theory": """<h2>1. Concept Overview</h2><p>Supervised Theory.</p>""",
    "ml-unit1-unsupervised-theory": """<h2>1. Concept Overview</h2><p>Unsupervised Theory.</p>""",
    "ml-unit1-rl-theory": """<h2>1. Concept Overview</h2><p>RL Theory.</p>""",
    "ml-unit1-applications": """<h2>1. Concept Overview</h2><p>Real World Apps.</p>""",
    "ml-unit1-summary": """<h2>1. Concept Overview</h2><p>Summary.</p>""",

    # ================= ML UNIT 2: SUPERVISED =================
    "ml-unit2-intro": """<h2>1. What is Supervised Learning?</h2><p>Labeled Data.</p>""",
    "ml-unit2-pre-overview": """
<h2>1. Why Preprocessing is Important</h2>
<p>Real-world data is often incomplete, inconsistent, and lacking in certain behaviors. <strong>"Garbage In, Garbage Out"</strong>: If you feed bad data to a model, you will get bad predictions.</p>
<h2>2. Data Quality Issues</h2>
<ul>
    <li><strong>Missing Values:</strong> Nulls or NaNs.</li>
    <li><strong>Outliers:</strong> Extreme values that skew analysis.</li>
    <li><strong>Inconsistent Formatting:</strong> Dates (DD/MM vs MM/DD), Case sensitivity.</li>
</ul>
<h2>3. Feature Scaling and Transformation</h2>
<p>Models like SVM and KNN use distance (Euclidean). If one feature has a range 0-1 and another 0-1000, the larger one will dominate. Scaling (0-1) or Standardization (Mean=0) fixes this.</p>
<h2>4. Preparing Data for ML Models</h2>
<p>The final dataset must be:</p>
<ul>
    <li><strong>Numeric:</strong> All text converted to numbers.</li>
    <li><strong>Scaled:</strong> All features on similar scale.</li>
    <li><strong>Split:</strong> Separated into X (Features) and y (Labels), and further into Train/Test sets.</li>
</ul>
""",
    "ml-unit2-mean-removal": """<h2>1. Mean Removal</h2><p>Standardization centers data at 0.</p>""",
    "ml-unit2-scaling": """<h2>1. Scaling</h2><p>Min-Max vs Z-Score.</p>""",
    "ml-unit2-binarization": """<h2>1. Binarization</h2><p>Thresholding.</p>""",
    "ml-unit2-linreg-basics": """<h2>1. Regression</h2><p>y=mx+b.</p>""",
    "ml-unit2-linreg-case": """<h2>1. Workflow</h2><p>Load -> Split -> Fit -> Predict.</p>""",
    "ml-unit2-cls-intro": """<h2>1. Classification</h2><p>Categories.</p>""",
    "ml-unit2-simple-cls": """<h2>1. Workflow</h2><p>Feature selection.</p>""",
    "ml-unit2-logistic": """<h2>1. Logistic</h2><p>Sigmoid.</p>""",
    "ml-unit2-naivebayes": """<h2>1. Naive Bayes</h2><p>Independence.</p>""",
    "ml-unit2-split": """<h2>1. Splitting</h2><p>Overfitting check.</p>""",
    "ml-unit2-accuracy": """<h2>1. Accuracy</h2><p>Cross-Validation.</p>""",
    "ml-unit2-conf-matrix": """<h2>1. Confusion Matrix</h2><p>TP/FP/TN/FN.</p>""",
    "ml-unit2-cls-report": """<h2>1. Report</h2><p>Precision/Recall.</p>""",
    "ml-unit2-predictive-intro": """<h2>1. Predictive</h2><p>Modeling future.</p>""",
    "ml-unit2-svm-theory": """<h2>1. SVM Theory</h2><p>Hyperplanes.</p>""",
    "ml-unit2-svm-types": """<h2>1. SVM Kernels</h2><p>Line vs Curve.</p>""",
    "ml-unit2-svm-confidence": """<h2>1. Confidence</h2><p>Distance from plane.</p>""",
    "ml-unit2-svm-case": """<h2>1. SVM Case</h2><p>Implementation.</p>""",
    "ml-unit2-summary": """<h2>1. Summary</h2><p>Recap.</p>""",

    # ================= UNIT 3: UNSUPERVISED LEARNING =================
    
    "ml-unit3-intro": """
<h2>1. What is Unsupervised Learning?</h2>
<p>Learning from <strong>Unlabeled Data</strong>. The system must discover information on its own.</p>
<h2>2. Uses</h2>
<p>Customer Segmentation, Anomaly Detection, Data Compression.</p>
<h2>3. Types</h2>
<ul>
    <li>Clustering (Grouping).</li>
    <li>Dimensionality Reduction (Simplifying).</li>
</ul>
""",
    "ml-unit3-clustering-overview": """
<h2>1. What is Clustering?</h2>
<p>Grouping data points so that points in the same group are more similar to each other than to those in other groups.</p>
<h2>2. Distance Measures</h2>
<ul>
    <li><strong>Euclidean:</strong> Straight line distance.</li>
    <li><strong>Manhattan:</strong> Grid-based traversal.</li>
</ul>
""",
    "ml-unit3-kmeans-theory": """
<h2>1. What is K-Means?</h2>
<p>A centroid-based algorithm.</p>
<h2>2. The Algorithm</h2>
<ol>
    <li>Initialize K centroids randomly.</li>
    <li>Assign points to nearest centroid.</li>
    <li>Recalculate centroids (Mean).</li>
    <li>Repeat until convergence.</li>
</ol>
<h2>3. Limitations</h2>
<p>You must choose K manually. Sensitive to outliers.</p>
""",
    "ml-unit3-kmeans-case": """
<h2>1. Workflow</h2>
<ol>
    <li><strong>Load Data:</strong> Mall Customers or Iris.</li>
    <li><strong>Elbow Method:</strong> Plot WCSS to find best K.</li>
    <li><strong>Fit:</strong> <code>kmeans = KMeans(n_clusters=5).fit(X)</code>.</li>
    <li><strong>Visualize:</strong> Scatter plot with colors.</li>
</ol>
""",
    "ml-unit3-img-compression": """
<h2>1. Vector Quantization</h2>
<p>Reducing the number of colors in an image.</p>
<h2>2. How?</h2>
<p>Use K-Means to find the top 16 dominant colors. Replace every pixel with the nearest dominant color.</p>
<h2>3. Result</h2>
<p>Drastically smaller file size with minimal visual loss.</p>
""",
    # REFINED: Mean Shift using Kernel Density
    "ml-unit3-meanshift": """
<h2>1. What is Mean Shift?</h2>
<p>A centroid-based algorithm that works by updating candidates for centroids to be the mean of the points within a given region.</p>
<h2>2. Kernel Density Concept</h2>
<p>It uses <strong>Kernel Density Estimation (KDE)</strong> to find the highest density of data points (modes).</p>
<h2>3. Pros & Cons</h2>
<ul>
    <li><strong>Pros:</strong> No need to select K (Clusters found automatically). Robust to outliers.</li>
    <li><strong>Cons:</strong> Computationally expensive (Slow).</li>
</ul>
""",
    "ml-unit3-agglomerative": """
<h2>1. Hierarchical Clustering</h2>
<p>Builds a hierarchy of clusters.</p>
<h2>2. Agglomerative (Bottom-Up)</h2>
<p>Start with N clusters (every point is a cluster). Merge nearest pair repeatedly until 1 giant cluster remains.</p>
<h2>3. Dendrogram</h2>
<p>A tree diagram that shows the order of merges.</p>
""",
    # REFINED: Comparison with When to Use Which
    "ml-unit3-comparison": """
<h2>1. K-Means vs Mean Shift vs Agglomerative</h2>
<ul>
    <li><strong>K-Means:</strong> Fast, Simple. Use when you know K and data is spherical.</li>
    <li><strong>Mean Shift:</strong> Slower. Use when you don't know K and clusters have irregular search.</li>
    <li><strong>Agglomerative:</strong> Hiearchical. Use for smaller datasets where visual tree (Dendrogram) is needed.</li>
</ul>
<h2>2. When to Use Which?</h2>
<p>Start with K-Means. If it fails on complex shapes, try Mean Shift (DBSCAN is also good). Use Hierarchical for taxonomy.</p>
""",
    "ml-unit3-summary": """
<h2>1. Key Takeaways</h2>
<p>Clustering is about similarity. K-Means is the workhorse.</p>
<h2>2. Interview Questions</h2>
<p><strong>Q:</strong> How to select K? <br><strong>A:</strong> Elbow Method or Silhouette Score.</p>
""",

    # 10. Semi-Supervised
    "ml-unit3-semisupervised": """
<h2>1. What is Semi-Supervised Learning?</h2>
<p>A hybrid approach that uses a <strong>small amount of labeled data</strong> and a <strong>large amount of unlabeled data</strong>.</p>
<h2>2. The Analogy</h2>
<p>Imagine a teacher solves 5 math problems (Labeled) and gives you 95 unsolved ones (Unlabeled). You use the logic from the 5 solved ones to try and solve the rest.</p>
<h2>3. Why use it?</h2>
<p>Labeling data is expensive (requires human experts, e.g., doctors for X-Rays). Unlabeled data is cheap. This method bridges the gap.</p>
<h2>4. Techniques</h2>
<p><strong>Self-Training:</strong> Train model on Labeled -> Predict Unlabeled -> Keep confident predictions as new Labeled data -> Retrain.</p>
""",

    # ================= UNIT 4: NLP =================

    "ml-unit4-intro": """
<h2>1. What is NLP?</h2>
<p><strong>Natural Language Processing</strong>: Teaching computers to understand human language.</p>
<h2>2. Applications</h2>
<p>Google Translate, Siri/Alexa, Spam Filter, Sentiment Analysis.</p>
""",
    "ml-unit4-pre-overview": """
<h2>1. Why Preprocess Text?</h2>
<p>Text is unstructured and messy (slang, emojis, caps).</p>
<h2>2. The Pipeline</h2>
<p>Clean -> Tokenize -> Stem/Lemmatize -> Vectorize -> Model.</p>
""",
    "ml-unit4-cleaning": """
<h2>1. Steps</h2>
<ul>
    <li><strong>Lowercasing:</strong> 'Hello' == 'hello'.</li>
    <li><strong>Stopword Removal:</strong> Removing 'the', 'is', 'at'.</li>
    <li><strong>Tokenization:</strong> Splitting sentences into words.</li>
</ul>
""",
    "ml-unit4-stemming": """
<h2>1. What is Stemming?</h2>
<p>Chopping off suffixes to find the root.</p>
<h2>2. Example</h2>
<p>Running, Ran, Runner -> <strong>Run</strong>.</p>
<h2>3. Porter Stemmer</h2>
<p>Fast but crude (Can produce 'Univers' instead of 'Universe').</p>
""",
    # REFINED: Lemmatization with POS Tagging
    "ml-unit4-lemmatization": """
<h2>1. What is Lemmatization?</h2>
<p>Using a dictionary and morphological analysis to find the root word (Lemma).</p>
<h2>2. POS Tagging</h2>
<p>It uses <strong>Part-of-Speech (POS)</strong> tagging to understand context (e.g., is 'meeting' a noun or a verb?) to lemmatize correctly.</p>
<h2>3. Difference from Stemming</h2>
<p>Lemmatization returns a real word. Stemming just chops the end.</p>
""",
    "ml-unit4-chunking": """
<h2>1. What is Chunking?</h2>
<p>Grouping tokens into meaningful phrases (Noun Phrases).</p>
<h2>2. Why?</h2>
<p>To extract entities like "New York City" as one unit, not three words.</p>
""",
    "ml-unit4-vectorization": """
<h2>1. Converting Text to Numbers</h2>
<p>Models only understand Math.</p>
<h2>2. Bag of Words (BoW)</h2>
<p>Counting word frequency. Simple but loses context.</p>
<h2>3. TF-IDF</h2>
<p><strong>Term Frequency - Inverse Document Frequency</strong>. Highlights unique/important words, ignores common ones.</p>
""",
    "ml-unit4-classifier": """
<h2>1. Workflow</h2>
<ol>
    <li><strong>Dataset:</strong> SMS Spam Collection.</li>
    <li><strong>Feature Extraction:</strong> TF-IDF.</li>
    <li><strong>Model:</strong> Naive Bayes (Standard for NLP).</li>
</ol>
""",
    "ml-unit4-case": """
<h2>1. Sentiment Analysis Python</h2>
<p>Building a movie review classifier.</p>
<h2>2. Steps</h2>
<p>Load Reviews -> Clean -> Vectorize -> Fit Model -> Predict "Positive/Negative".</p>
""",
    "ml-unit4-summary": """
<h2>1. Key Takeaways</h2>
<p>Preprocessing is 80% of NLP. TF-IDF > CountVectorizer usually.</p>
""",

    # ================= UNIT 5: COMPUTER VISION =================

    "ml-unit5-intro": """
<h2>1. What is Computer Vision?</h2>
<p>Enabling computers to "see" and interpret images.</p>
<h2>2. Why OpenCV?</h2>
<p>Open Source Computer Vision Library. The industry standard for real-time vision.</p>
""",
    "ml-unit5-opencv-basics": """
<h2>1. Installation</h2>
<p><code>pip install opencv-python</code></p>
<h2>2. Reading Images</h2>
<p>Images are numpy arrays of pixels (Height, Width, Channels).</p>
""",
    "ml-unit5-haar-theory": """
<h2>1. What is a Haar Cascade?</h2>
<p>An ML-based approach where a cascade function is trained from a lot of positive and negative images.</p>
<h2>2. Viola-Jones Algorithm</h2>
<p>Fast feature extraction using rectangular regions (Haar features).</p>
""",
    "ml-unit5-face-detect": """
<h2>1. How it works</h2>
<p>We load a pre-trained XML file (frontal face). We pass our image to the detector.</p>
<h2>2. Bounding Box</h2>
<p>The detector returns (x, y, w, h). We draw a rectangle using these coordinates.</p>
""",
    "ml-unit5-features": """
<h2>1. ROI (Region of Interest)</h2>
<p>Once a face is found, crop it. Search for Eyes/Nose ONLY inside that face box. This saves CPU.</p>
""",
    "ml-unit5-obj-img": """
<h2>1. Tuning Parameters</h2>
<ul>
    <li><strong>scaleFactor:</strong> How much image size is reduced at each scale.</li>
    <li><strong>minNeighbors:</strong> Quality check (higher = fewer detections but better quality).</li>
</ul>
""",
    "ml-unit5-obj-vid": """
<h2>1. Webcam Access</h2>
<p><code>cv2.VideoCapture(0)</code> accesses the default camera.</p>
<h2>2. Real-Time Loop</h2>
<p>Read Frame -> Process -> Display -> Wait for 'q' to quit.</p>
""",
    "ml-unit5-tracking": """
<h2>1. Tracking Logic</h2>
<p>Simply running detection on every frame works for fast CPUs. For slower ones, detect once every 30 frames and 'track' in between.</p>
""",
    "ml-unit5-pupil": """
<h2>1. Advanced Detection</h2>
<p>Haar Finds the eye. Then use <strong>Thresholding</strong> to find the darkest blob (pupil).</p>
<h2>2. Contours</h2>
<p><code>cv2.findContours</code> outlines the shape of the pupil.</p>
""",
    "ml-unit5-case": """
<h2>1. The Application</h2>
<p><strong>Drowsiness Detector:</strong></p>
<p>Detect Eyes -> Check if closed for 3 seconds -> Sound Alarm.</p>
""",
    "ml-unit5-summary": """
<h2>1. Summary</h2>
<p>CV is powerful. Haar Cascades are old but fast. Deep Learning (YOLO) is modern but heavy.</p>
"""
}

def expand_tutorials():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        tutorials = json.load(f)

    count = 0
    for t in tutorials:
        t_id = t.get('id', '')
        if t_id in CONTENT_MAP:
            t['explanation'] = CONTENT_MAP[t_id]
            count += 1

    with open(JSON_PATH, 'w') as f:
        json.dump(tutorials, f, indent=4)
        
    print(f"Successfully expanded {count} tutorials to Textbook Style.")

if __name__ == "__main__":
    expand_tutorials()
