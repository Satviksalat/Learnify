
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR ML UNIT 3
ML_UNIT_3_DATA = {
    "unit": "ML Unit 3: Unsupervised Learning",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Unsupervised Learning.", "answer": "A type of machine learning where the model learns from unlabeled data to find hidden patterns."},
            {"question": "What is Clustering?", "answer": "The task of grouping a set of objects in such a way that objects in the same group are more similar."},
            {"question": "Define K-Means Clustering.", "answer": "An iterative algorithm that partitions data into K distinct, non-overlapping clusters."},
            {"question": "What is a Centroid?", "answer": "The center point of a cluster, often calculated as the mean of all data points in that cluster."},
            {"question": "Define Dimensionality Reduction.", "answer": "The process of reducing the number of random variables under consideration (features)."},
            {"question": "What is PCA?", "answer": "Principal Component Analysis: A technique to emphasize variation and bring out strong patterns in a dataset."},
            {"question": "What is a Scree Plot?", "answer": "A line plot used to determine the number of factors to retain in PCA."},
            {"question": "Define Eigenvector.", "answer": "A vector that does not change direction during a linear transformation, only its magnitude."},
            {"question": "Define Eigenvalue.", "answer": "A scalar that indicates the magnitude of the eigenvector."},
            {"question": "What is Hierarchical Clustering?", "answer": "An algorithm that groups objects into a hierarchy or tree-like structure of clusters."},
            {"question": "What is a Dendrogram?", "answer": "A tree diagram used to visualize the arrangement of clusters produced by hierarchical clustering."},
            {"question": "Define Agglomerative Clustering.", "answer": "A 'bottom-up' approach where each observation starts in its own cluster, and pairs are merged."},
            {"question": "Define Divisive Clustering.", "answer": "A 'top-down' approach where all observations start in one cluster, and splits are performed recursively."},
            {"question": "What is Vector Quantization?", "answer": "A classical quantization technique from signal processing that allows the modeling of probability density functions."},
            {"question": "What is Mean Shift?", "answer": "A non-parametric clustering algorithm that does not require defining the number of clusters in advance."},
            {"question": "Define Anomaly Detection.", "answer": "The identification of rare items, events, or observations which raise suspicions by differing from the majority."},
            {"question": "What is Association Rule Mining?", "answer": "A procedure to find frequent patterns, correlations, or causal structures in datasets."},
            {"question": "Define the Apriori Principle.", "answer": "If an itemset is frequent, then all of its subsets must also be frequent."},
            {"question": "What is Silhouette Score?", "answer": "A metric to evaluate clustering quality, measuring how similar a point is to its own cluster vs others."},
            {"question": "What is Adjusted Rand Index?", "answer": "A measure of the similarity between two data clusterings, corrected for chance."}
        ],
        "Part B (2-Marks)": [
            {"question": "Supervised vs Unsupervised Learning.", "answer": "• Supervised: Labeled data, Goal is prediction (Regression/Classification).\n• Unsupervised: Unlabeled data, Goal is structure discovery (Clustering)."},
            {"question": "List 4 Applications of Clustering.", "answer": "• Customer Segmentation\n• Image Compression\n• Document Classification\n• Anomaly Detection"},
            {"question": "Euclidean vs Manhattan Distance.", "answer": "• Euclidean: Straight-line distance between two points (L2 Norm).\n• Manhattan: Sum of absolute differences (L1 Norm) / Grid-like path."},
            {"question": "What is the Elbow Method?", "answer": "• A heuristic to find the optimal number of clusters (K).\n• Plots WCSS vs K; the 'elbow' point indicates the best K."},
            {"question": "Explain the Curse of Dimensionality.", "answer": "• As features increase, data becomes sparse, making distance metrics less meaningful.\n• Requires more data to generalize accurately."},
            {"question": "Feature Selection vs Extraction.", "answer": "• Selection: Choosing a subset of relevant features existing in the data.\n• Extraction: Creating new, smaller set of features from the original ones (e.g., PCA)."},
            {"question": "Goal of PCA.", "answer": "• To reduce the dimensionality of large datasets.\n• To preserve as much statistical information (variance) as possible."},
            {"question": "What is Inertia in K-Means?", "answer": "• The sum of squared distances of samples to their closest cluster center.\n• Lower inertia means tighter clusters (better clustering)."},
            {"question": "Steps in Hierarchical Clustering.", "answer": "• Calculate distance matrix.\n• Merge two closest clusters.\n• Update distance matrix. Repeat until one cluster remains."},
            {"question": "Single vs Complete Linkage.", "answer": "• Single: Distance between closest pair of points in two clusters.\n• Complete: Distance between farthest pair of points in two clusters."},
            {"question": "DBSCAN vs K-Means.", "answer": "• K-Means: Needs 'K' predefined. Assumes spherical clusters.\n• DBSCAN: Density-based. Finds arbitrary shapes. Handles noise better."},
            {"question": "Use cases for Anomaly Detection.", "answer": "• Credit Card Fraud Detection.\n• Network Intrusion Detection."},
            {"question": "What is Market Basket Analysis?", "answer": "• Analyzing customer purchasing habits to find products bought together.\n• Example: Bread and Butter are often bought together."},
            {"question": "Logic of Image Compression via Clustering.", "answer": "• Reduce number of unique colors in an image.\n• Group similar colors into 'K' clusters and replace pixels with centroid color."},
            {"question": "What is Customer Segmentation?", "answer": "• Grouping customers based on behavior, demographics, or purchase history.\n• Helps in targeted marketing strategies."}
        ],
        "Part C (3-Marks)": [
            {"question": "Steps in K-Means Algorithm.", "answer": "1. **Definition:** An iterative clustering algorithm.\n2. **Explanation:** 1. Initialize K centroids randomly. 2. Assign points to nearest centroid. 3. Recalculate centroids. 4. Repeat until convergence.\n3. **Example:** Grouping T-shirts into Small, Medium, Large sizes."},
            {"question": "How to choose K (Elbow Method).", "answer": "1. **Definition:** A visual method to determine the optimal number of clusters.\n2. **Explanation:** Run K-means for range of K (1-10). Plot Inertia (Y) vs K (X). Look for the 'bend' or elbow.\n3. **Example:** If graph bends at K=3, use 3 clusters."},
            {"question": "Hierarchical Clustering Algorithm.", "answer": "1. **Definition:** Builds a tree of clusters.\n2. **Explanation:** Calculates distance between all points. Merges closest pairs iteratively until a single tree (Dendrogram) is formed.\n3. **Example:** Biological taxonomy (Kingdom -> Species)."},
            {"question": "Single Linkage Explained.", "answer": "1. **Definition:** A method to calculate distance between clusters.\n2. **Explanation:** Uses the minimum distance between any single point in Cluster A and any single point in Cluster B.\n3. **Example:** Can result in long, 'chain-like' clusters."},
            {"question": "Complete Linkage Explained.", "answer": "1. **Definition:** A method to calculate distance between clusters.\n2. **Explanation:** Uses the maximum distance between points in Cluster A and Cluster B. Tends to find compact clusters.\n3. **Example:** Useful when clusters are distinct and spherical."},
            {"question": "Steps in PCA.", "answer": "1. **Definition:** Principal Component Analysis workflow.\n2. **Explanation:** 1. Standardize data. 2. Compute Covariance Matrix. 3. Compute Eigenvectors/values. 4. Sort and select top K.\n3. **Example:** Reducing 100 facial features to 10 'Eigenfaces'."},
            {"question": "Image Compression via K-Means.", "answer": "1. **Definition:** Reducing image size by reducing color palette.\n2. **Explanation:** Treat pixels as data points (R,G,B). Cluster into K colors. Replace all pixels with their cluster center.\n3. **Example:** Compressing a 16-million color photo to 256 colors."},
            {"question": "Mean Shift Clustering Logic.", "answer": "1. **Definition:** A centroid-based algorithm that works by updating candidates for potential centroids to be the mean of the points within a given region.\n2. **Explanation:** Like climbing a hill of density. Points move towards high-density areas.\n3. **Example:** Used in Computer Vision for object tracking."},
            {"question": "Agglomerative vs Divisive Clustering.", "answer": "1. **Definition:** Two approaches to Hierarchical Clustering.\n2. **Explanation:** Agglomerative: Bottom-Up (Start with N clusters, merge to 1). Divisive: Top-Down (Start with 1, split to N).\n3. **Example:** Agglomerative is more common in software packages."},
            {"question": "Silhouette Analysis.", "answer": "1. **Definition:** A graphical tool to measure how tightly grouped samples are in clusters.\n2. **Explanation:** Range [-1, 1]. +1 = Far from neighbors (Good). 0 = On border. -1 = Wrong cluster.\n3. **Example:** Average score of 0.7 indicates strong structure."},
            {"question": "Advantages of Unsupervised Learning.", "answer": "1. **Definition:** Benefits of learning without labels.\n2. **Explanation:** No need for expensive manual labeling. Can discover hidden, unknown patterns in data.\n3. **Example:** Finding a new customer segment no one knew existed."},
            {"question": "Disadvantages of Unsupervised Learning.", "answer": "1. **Definition:** Limitations of learning without labels.\n2. **Explanation:** Hard to validate results (no Ground Truth). Less accurate than supervised learning for specific tasks.\n3. **Example:** A clustering model might group 'Wolves' and 'Dogs' together incorrectly."},
            {"question": "Distance Metrics.", "answer": "1. **Definition:** Mathematical formulas to measure similarity.\n2. **Explanation:** Euclidean (L2): Shortest path. Manhattan (L1): City block path. Cosine: Angle between vectors.\n3. **Example:** Cosine distance is used for Text Similarity."},
            {"question": "Association Rule Mining Metrics.", "answer": "1. **Definition:** Measures of interestingness for rules.\n2. **Explanation:** Support (Frequency), Confidence (Reliability), Lift (Correlation > random chance).\n3. **Example:** High Lift means X and Y are dependent."},
            {"question": "Semi-Supervised Learning Logic.", "answer": "1. **Definition:** Uses a small amount of labeled data and lots of unlabeled data.\n2. **Explanation:** Propagates labels from the known points to nearby unknown points to expand the training set.\n3. **Example:** Google Photos labeling a few faces and it recognizing the rest."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Clustering.",
                "answer": "1. **Definition:**\n   An unsupervised learning task that groups a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.\n\n2. **Goal:**\n   To discover inherent structures or patterns in data without using predefined labels.\n\n3. **Core Concept:**\n   Intra-cluster similarity (high) and Inter-cluster similarity (low).\n\n4. **Technique / Method:**\n   Algorithms like K-Means (Centroid), Hierarchical (Connectivity), and DBSCAN (Density).\n\n5. **Applications:**\n   Market Segmentation, Social Network Analysis, Search Result Grouping, Medical Imaging."
            },
            {
                "question": "Detailed Explanation of K-Means Clustering.",
                "answer": "1. **Definition:**\n   One of the simplest and widely used unsupervised learning algorithms that solves the clustering problem.\n\n2. **Goal:**\n   To partition 'n' observations into 'k' clusters in which each observation belongs to the cluster with the nearest mean.\n\n3. **Core Concept:**\n   Iterative refinement. Assign points to nearest centroid -> Move centroid to mean of points -> Repeat.\n\n4. **Technique / Method:**\n   Requires defining 'K' beforehand. Uses Euclidean distance. Sensitive to initialization (K-Means++).\n\n5. **Applications:**\n   Document Classification, Customer Profiling, Ride-share demand identification."
            },
            {
                "question": "Detailed Explanation of Hierarchical Clustering.",
                "answer": "1. **Definition:**\n   A method of cluster analysis which seeks to build a hierarchy of clusters.\n\n2. **Goal:**\n   To treat data as a tree-like structure (Dendrogram) rather than flat clusters.\n\n3. **Core Concept:**\n   Agglomerative (Merge closest pairs) or Divisive (Split farthest points).\n\n4. **Technique / Method:**\n   Does not require 'K'. Uses Linkage criteria (Single, Complete, Average, Ward) to measure distance.\n\n5. **Applications:**\n   Gene Expression Analysis (Bioinformatics), Taxonomy creation, Organization structures."
            },
            {
                "question": "Detailed Explanation of Principal Component Analysis (PCA).",
                "answer": "1. **Definition:**\n   A statistical procedure that uses an orthogonal transformation to convert correlations into a set of values of linearly uncorrelated variables called principal components.\n\n2. **Goal:**\n   Dimensionality Reduction. To reduce the number of variables while preserving the most information (variance).\n\n3. **Core Concept:**\n   Project data onto new axes (Eigenvectors) where the data is most spread out (max variance).\n\n4. **Technique / Method:**\n   Compute Covariance Matrix -> Calculate Eigenvalues/Eigenvectors -> Select top components.\n\n5. **Applications:**\n   Face Recognition (Eigenfaces), Image Compression, Visualization of high-dimensional data."
            },
            {
                "question": "Detailed Explanation of Mean Shift Clustering.",
                "answer": "1. **Definition:**\n   A non-parametric feature-space analysis technique for locating the maxima of a density function.\n\n2. **Goal:**\n   To discover clusters of arbitrary shapes without defining the number of clusters 'K' in advance.\n\n3. **Core Concept:**\n   Sliding Window approach. A window moves towards higher density areas until it finds the peak (centroid).\n\n4. **Technique / Method:**\n   Define window size (Bandwidth). Shift window mean iteratively. Points converging to same peak belong to same cluster.\n\n5. **Applications:**\n   Computer Vision (Image Segmentation), Object Tracking in video."
            },
            {
                "question": "Detailed Explanation of Image Compression using Vector Quantization.",
                "answer": "1. **Definition:**\n   A lossy data compression method used in image processing based on the principle of block coding.\n\n2. **Goal:**\n   To reduce the storage space of an image by reducing the number of colors while identifying regions.\n\n3. **Core Concept:**\n   Clustering colors. If an image has 1000 colors, cluster them into 16 groups. Replace all 1000 with the 16 centroids.\n\n4. **Technique / Method:**\n   Use K-Means to find 'K' representative colors (Codebook). Map every pixel to the nearest Codebook index.\n\n5. **Applications:**\n   GIF format, JPEG compression, Video conferencing bandwidth reduction."
            },
            {
                "question": "Comparative Study of Clustering Algorithms.",
                "answer": "1. **Definition:**\n   Evaluating different clustering methods to choose the best one for a specific dataset.\n\n2. **Goal:**\n   To balance speed, accuracy, and ability to handle noise and shape.\n\n3. **Core Concept:**\n   K-Means (Fast, Spherical). Hierarchical (Interpretability, Slow). DBSCAN (Noise-resistant, Arb shape).\n\n4. **Technique / Method:**\n   Use metrics like Silhouette Score, Davies-Bouldin Index. Test stability.\n\n5. **Applications:**\n   Choosing DBSCAN for geospatial data (maps) vs K-Means for simple customer grouping."
            },
            {
                "question": "Detailed Explanation of Anomaly Detection.",
                "answer": "1. **Definition:**\n   The process of identifying unexpected items or events in data sets, which differ from the norm.\n\n2. **Goal:**\n   To detect rare events that could identify problems like bank fraud, medical problems, or structural defects.\n\n3. **Core Concept:**\n   Model the 'Normal' behavior. Anything that deviates significantly (Distance/Density) is an anomaly.\n\n4. **Technique / Method:**\n   Isolation Forest, One-Class SVM, or Density-based checks. Compute outlier score.\n\n5. **Applications:**\n   Intrusion Detection Systems (Cybersecurity), Manufacturing Fault Detection."
            },
            {
                "question": "Detailed Explanation of Association Rule Mining.",
                "answer": "1. **Definition:**\n   A rule-based machine learning method for discovering interesting relations between variables in large databases.\n\n2. **Goal:**\n   To find strong rules discovered in databases using some measures of interestingness.\n\n3. **Core Concept:**\n   If-Then rules. {Diapers} -> {Beer}. Finding items that co-occur frequently.\n\n4. **Technique / Method:**\n   Apriori Algorithm (Frequent Itemset generation). Metrics: Support, Confidence, Lift.\n\n5. **Applications:**\n   Market Basket Analysis (Retail), Web Usage Mining, Bioinformatics."
            },
            {
                "question": "Detailed Explanation of Semi-Supervised Learning.",
                "answer": "1. **Definition:**\n   A class of machine learning techniques that makes use of both labeled and unlabeled data for training.\n\n2. **Goal:**\n   To overcome the labeling bottleneck (expensive/slow) by leveraging abundant cheap unlabeled data.\n\n3. **Core Concept:**\n   Continuity Assumption: Points close to each other are likely to share a label.\n\n4. **Technique / Method:**\n   Self-Training: Train on labeled -> Predict unlabeled -> Add confident predictions to training set -> Repeat.\n\n5. **Applications:**\n   Speech Analysis (few transcribed hours), Web Content Classification (billions of pages)."
            }
        ]
    }
}

def populate_ml_unit3():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Unit 3: Unsupervised Learning"
        data = [u for u in data if "Unit 3: Unsupervised Learning" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(ML_UNIT_3_DATA)
        print("Successfully replaced ML Unit 3 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_ml_unit3()
