
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR ML UNIT 4
ML_UNIT_4_DATA = {
    "unit": "ML Unit 4: Natural Language Processing",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Natural Language Processing (NLP).", "answer": "A branch of AI that gives computers the ability to understand, interpret, and generate human language."},
            {"question": "What is Tokenization?", "answer": "The process of breaking down text into smaller units called tokens (words or sentences)."},
            {"question": "Define Stemming.", "answer": "The process of reducing words to their root/base form (e.g., 'Running' -> 'Run')."},
            {"question": "Define Lemmatization.", "answer": "Reducing words to their dictionary root form (Lemma) using vocabulary analysis (e.g., 'Better' -> 'Good')."},
            {"question": "What are Stop Words?", "answer": "Common words (like 'is', 'the', 'and') that carry little meaning and are often removed."},
            {"question": "What is Bag of Words (BoW)?", "answer": "A representation of text that describes the occurrence of words within a document, ignoring order."},
            {"question": "Define TF-IDF.", "answer": "Term Frequency-Inverse Document Frequency. A statistic to evaluate how important a word is to a document."},
            {"question": "What is a Corpus?", "answer": "A large and structured set of texts (datasets) used for training NLP models."},
            {"question": "Define N-Gram.", "answer": "A contiguous sequence of 'n' items (words/chars) from a given sample of text."},
            {"question": "What is POS Tagging?", "answer": "Part-of-Speech Tagging: Assigning grammatical tags (Noun, Verb, Adj) to each word in a generic text."},
            {"question": "Define Named Entity Recognition (NER).", "answer": "Identifying and classifying key entities in text into categories like Names, Dates, Locations."},
            {"question": "What is Word Embedding?", "answer": "A representation of words where words with similar meanings have a similar numerical representation."},
            {"question": "Define Sentiment Analysis.", "answer": "The process of determining the emotional tone behind a series of words (Positive/Negative/Neutral)."},
            {"question": "What uses Word2Vec?", "answer": "A technique to produce word embeddings using neural networks to learn word associations."},
            {"question": "Define Text Preprocessing.", "answer": "The step of cleaning and preparing raw text data for NLP tasks (Lowercasing, Noise Removal)."},
            {"question": "What is Chunking?", "answer": "Grouping individual tokens into meaningful phrases (like Noun Phrases) based on POS tags."},
            {"question": "Syntax vs Semantics.", "answer": "Syntax refers to the grammatical structure. Semantics refers to the actual meaning of the text."},
            {"question": "What is a Chatbot?", "answer": "A software application used to conduct an on-line chat conversation via text or text-to-speech."},
            {"question": "Define Latent Semantic Analysis (LSA).", "answer": "A technique to analyze relationships between a set of documents and the terms they contain."},
            {"question": "What is the Turing Test?", "answer": "A test of a machine's ability to exhibit intelligent behavior equivalent to typical human responses."}
        ],
        "Part B (2-Marks)": [
            {"question": "Stemming vs Lemmatization.", "answer": "• Stemming: Fast, chops off ends, may result in non-words (Caring -> Car).\n• Lemmatization: Slow, uses dictionary, results in actual words (Caring -> Care)."},
            {"question": "Tokenization vs Segmentation.", "answer": "• Tokenization: Splitting text into words.\n• Segmentation: Splitting text into sentences."},
            {"question": "Bag of Words vs TF-IDF.", "answer": "• BoW: Counts frequency. Biased towards common words.\n• TF-IDF: Weighs frequency by rarity across documents. Penalizes common words."},
            {"question": "List 4 Applications of NLP.", "answer": "• Machine Translation (Google Translate)\n• Spam Filtering\n• Voice Assistants (Siri)\n• Sentiment Analysis"},
            {"question": "Steps in Text Preprocessing.", "answer": "• 1. Cleaning (Remove HTML/URLs). 2. Tokenization.\n• 3. Stop Word Removal. 4. Stemming/Lemmatization."},
            {"question": "Role of Stop Word Removal.", "answer": "• Reduces dataset size.\n• Removes noise (meaningless words) to focus on key keywords."},
            {"question": "What is CountVectorizer?", "answer": "• A tool to convert a collection of text documents to a matrix of token counts.\n• It implements the Bag of Words method."},
            {"question": "Challenges in NLP.", "answer": "• Ambiguity (Words having multiple meanings).\n• Sarcasm and Irony detection."},
            {"question": "Rule-based vs Statistical NLP.", "answer": "• Rule-based: Uses handcrafted linguistic rules (Grammar).\n• Statistical: Uses Machine Learning algorithms to learn patterns from data."},
            {"question": "One-Hot Encoding in NLP.", "answer": "• Representing each word as a binary vector.\n• Sparse and high-dimensional; inefficient for large vocabularies."},
            {"question": "Unigram vs Bigram.", "answer": "• Unigram: Single words ('New', 'York').\n• Bigram: Pairs of consecutive words ('New York'). Preserves some context."},
            {"question": "Why convert text to numbers?", "answer": "• Machines cannot understand raw text strings.\n• Algorithms require numerical input (vectors) to perform calculations."},
            {"question": "Syntactic vs Semantic Ambiguity.", "answer": "• Syntactic: Confusion in sentence structure ('I saw the man with the telescope').\n• Semantic: Confusion in word meaning ('Bank' of river vs 'Bank' of money)."},
            {"question": "What is a Document-Term Matrix (DTM)?", "answer": "• A mathematical matrix that describes the frequency of terms that occur in a collection of documents.\n• Rows = Documents, Columns = Words."},
            {"question": "Cosine Similarity in NLP.", "answer": "• A metric used to measure how similar two documents are.\n• Calculates the cosine of the angle between two vectors."}
        ],
        "Part C (3-Marks)": [
            {"question": "Text Preprocessing Workflow.", "answer": "1. **Definition:** Validating and cleaning text for analysis.\n2. **Explanation:** Raw Text -> Lowercase -> Remove Punctuation -> Tokenize -> Remove Stopwords -> Stem.\n3. **Example:** 'Hi! There' -> 'hi there' -> ['hi', 'there']."},
            {"question": "Explain TF-IDF Calculation.", "answer": "1. **Definition:** Statistic for word importance.\n2. **Explanation:** TF = (Count of word in doc) / (Total words). IDF = log(Total docs / Docs with word). Score = TF * IDF.\n3. **Example:** Rare word 'Quantum' has high IDF. Common word 'The' has low IDF."},
            {"question": "Explain Sentiment Analysis.", "answer": "1. **Definition:** Mining opinions from text.\n2. **Explanation:** Classifies text as Positive, Negative, or Neutral based on polarity scores of words.\n3. **Example:** 'I love this movie' -> Positive. 'It was boring' -> Negative."},
            {"question": "Goal of Named Entity Recognition.", "answer": "1. **Definition:** Information extraction task.\n2. **Explanation:** To locate and classify named entities in unstructured text into pre-defined categories.\n3. **Example:** 'Steve Jobs (Person) founded Apple (Org) in 1976 (Date).'"},
            {"question": "CountVectorizer vs TfidfVectorizer.", "answer": "1. **Definition:** Two methods to vectorizing text.\n2. **Explanation:** CountVec just counts word occurrences (Integer counts). TfidfVec normalizes counts by rarity (Float scores).\n3. **Example:** Tfidf is better for Information Retrieval/Search."},
            {"question": "Usage of N-Grams.", "answer": "1. **Definition:** Sequences of N words.\n2. **Explanation:** Captures context that single words miss. 'Not' + 'Good' is different from 'Not' and 'Good'.\n3. **Example:** 'Not Good' (Bigram) carries negative sentiment."},
            {"question": "POS Tagging Explanation.", "answer": "1. **Definition:** Labeling words with their part of speech.\n2. **Explanation:** Knowing if a word is a Noun or Verb changes its meaning. Helps in Lemmatization.\n3. **Example:** 'Book a flight' (Verb) vs 'Read a book' (Noun)."},
            {"question": "Chunking vs Chinking.", "answer": "1. **Definition:** Shallow parsing techniques.\n2. **Explanation:** Chunking: Grouping tokens into phrases (e.g., Noun Phrases). Chinking: Excluding tokens from a chunk.\n3. **Example:** Chunk: {The small cat} {sat}. Chink: Remove verbs from chunk."},
            {"question": "Word2Vec Concept.", "answer": "1. **Definition:** A neural network model to create word embeddings.\n2. **Explanation:** Words appearing in similar contexts are mapped close together in vector space.\n3. **Example:** King - Man + Woman = Queen."},
            {"question": "Chatbot Architecture Basics.", "answer": "1. **Definition:** Components of a conversational agent.\n2. **Explanation:** User Input -> NLP (Intent Recognition) -> Dialog Manager (Logic) -> NLG (Response Gen) -> Output.\n3. **Example:** User says 'Hello', Bot maps to 'Greeting' intent."},
            {"question": "Ambiguity in NLP.", "answer": "1. **Definition:** Uncertainty of meaning.\n2. **Explanation:** Lexical (Word meanings), Syntactic (Sentence structure), Referential (Pronouns).\n3. **Example:** 'The chicken is ready to eat' (Is the chicken eating or being eaten?)."},
            {"question": "Text Classification Steps.", "answer": "1. **Definition:** Assigning tags to text.\n2. **Explanation:** 1. Vectorize Text (TF-IDF). 2. Train Classifier (Naive Bayes). 3. Predict Label.\n3. **Example:** Categorizing email into 'Work', 'Social', 'Spam'."},
            {"question": "Topic Modeling (LDA).", "answer": "1. **Definition:** Unsupervised method to find topics in text.\n2. **Explanation:** Latent Dirichlet Allocation. Assumes docs are mixtures of topics and topics are mixtures of words.\n3. **Example:** Finding 'Sports' and 'Politics' topics in a newspaper corpus."},
            {"question": "Advantages of Stemming.", "answer": "1. **Definition:** Pros of suffix stripping.\n2. **Explanation:** Reduces vocabulary size significantly. Very fast to compute. Good for search engines.\n3. **Example:** 'Fish', 'Fishing', 'Fished' all map to 'Fish'."},
            {"question": "Disadvantages of BoW.", "answer": "1. **Definition:** Cons of Bag of Words.\n2. **Explanation:** Ignores word order (context). Results in very sparse matrices (mostly zeros).\n3. **Example:** 'Man eats Dog' and 'Dog eats Man' have exact same BoW vector."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of NLP (Introduction).",
                "answer": "1. **Definition:**\n   The discipline of building machines that can understand and respond to natural human languages.\n\n2. **Goal:**\n   To bridge the gap between human communication (ambiguous/complex) and computer understanding (binary/logical).\n\n3. **Core Concept:**\n   Computational Linguistics + Machine Learning. Processing text to extract meaning and intent.\n\n4. **Technique / Method:**\n   Pipeline: Tokenization -> Preprocessing -> Feature Extraction -> Modeling (ML/DL).\n\n5. **Applications:**\n   Virtual Assistants (Alexa), Spam Filters, Auto-Correct, Language Translation."
            },
            {
                "question": "Detailed Text Preprocessing Pipeline.",
                "answer": "1. **Definition:**\n   The series of steps taken to clean raw text and make it suitable for machine learning models.\n\n2. **Goal:**\n   To remove noise and standardize the text to reduce the dimensionality of the problem.\n\n3. **Core Concept:**\n   Garbage In, Garbage Out. Cleaner text leads to better models.\n\n4. **Technique / Method:**\n   1. Start with cleaning (regex). 2. Tokenize. 3. Stopword removal. 4. Stem/Lemmatize. 5. Vectorize.\n\n5. **Applications:**\n   Required for ANY NLP task, from simple word clouds to complex sentiment analysis."
            },
            {
                "question": "Comparison: Stemming vs Lemmatization.",
                "answer": "1. **Definition:**\n   Two techniques for text normalization that reduce words to their base forms.\n\n2. **Goal:**\n   To map different forms of a word (running, runs, ran) to a single token to simplify analysis.\n\n3. **Core Concept:**\n   Stemming is heuristic (rules). Lemmatization is linguistic (dictionary).\n\n4. **Technique / Method:**\n   Stemming: Chops suffix (Porter Algorithm). Lemmatization: Look up lemma based on POS (WordNet).\n\n5. **Applications:**\n   Search engines use Stemming for speed. Chatbots use Lemmatization for accuracy."
            },
            {
                "question": "Bag of Words (BoW) vs TF-IDF.",
                "answer": "1. **Definition:**\n   Two common feature extraction techniques to convert text into numerical vectors.\n\n2. **Goal:**\n   To represent the content of a document mathematically for ML algorithms.\n\n3. **Core Concept:**\n   BoW counts frequency (Quantity). TF-IDF measures relevance (Quality/Uniqueness).\n\n4. **Technique / Method:**\n   BoW = Count matrix. TF-IDF = TF * log(N/df). TF-IDF penalizes common words like 'the'.\n\n5. **Applications:**\n   BoW for simple Classification. TF-IDF for Search Engines and Keyword Extraction."
            },
            {
                "question": "Detailed Explanation of Sentiment Analysis.",
                "answer": "1. **Definition:**\n   The computational study of people's opinions, sentiments, emotions, and attitudes.\n\n2. **Goal:**\n   To classify the polarity of a given text as Positive, Negative, or Neutral.\n\n3. **Core Concept:**\n   Opinion Mining. Identifying subjective information in source materials.\n\n4. **Technique / Method:**\n   Lexicon-based (Dictionary of positive/negative words) or ML-based (Train Naive Bayes on reviews).\n\n5. **Applications:**\n   Social Media Monitoring (Brand reputation), Customer Feedback Analysis, Product Reviews."
            },
            {
                "question": "Detailed Explanation of Named Entity Recognition (NER).",
                "answer": "1. **Definition:**\n   A subtask of information extraction that identifies named entities in unstructured text.\n\n2. **Goal:**\n   To locate and classify entities into predefined categories (Person, Org, Location, Date).\n\n3. **Core Concept:**\n   Turning unstructured text into structured data points.\n\n4. **Technique / Method:**\n   Uses Statistical models (CRFs) or Deep Learning (LSTM/BERT) to predict entity tags.\n\n5. **Applications:**\n   News Content Classification, Customer Support (extracting Order IDs), Bio-medical text mining."
            },
            {
                "question": "Detailed Explanation of Word Embeddings (Word2Vec).",
                "answer": "1. **Definition:**\n   A technique where words are encoded as real-valued vectors in a dense, low-dimensional space.\n\n2. **Goal:**\n   To capture semantic meanings and relationships between words (e.g., King/Queen).\n\n3. **Core Concept:**\n   Distributional Hypothesis: Words that appear in the same context have similar meanings.\n\n4. **Technique / Method:**\n   Word2Vec (CBOW or Skip-Gram). Neural network learns to predict context words from target words.\n\n5. **Applications:**\n   Recommendation Systems, Semantic Search, Analogy detection (Paris is to France as Tokyo is to Japan)."
            },
            {
                "question": "Real-World Applications of NLP.",
                "answer": "1. **Definition:**\n   Practical implementations of language processing in industry and daily life.\n\n2. **Goal:**\n   To automate communication-heavy tasks and extract insights from textual data.\n\n3. **Core Concept:**\n   Automating the read/understand/reply loop.\n\n4. **Technique / Method:**\n   Translation (Seq2Seq), Summarization (Transformers), Speech-to-Text (ASR).\n\n5. **Applications:**\n   Google Translate, Email Spam filters, Siri/Alexa, Chatbots, Stock Market Prediction (News)."
            },
            {
                "question": "Challenges in Natural Language Processing.",
                "answer": "1. **Definition:**\n   The difficulties inherent in processing human language due to its complexity and irregularity.\n\n2. **Goal:**\n   To build robust systems that can handle real-world, messy language.\n\n3. **Core Concept:**\n   Human language is Ambiguous, Context-dependent, and constantly Evolving (Slang).\n\n4. **Technique / Method:**\n   Issues: Polysemy (Ambiguity), Sarcasm, idioms, spelling errors, and lack of training data for low-resource languages.\n\n5. **Applications:**\n   Why Siri sometimes misunderstands you or why translation fails on poetry."
            },
            {
                "question": "Steps to Build a Text Classifier.",
                "answer": "1. **Definition:**\n   The end-to-end process of creating a model to categorize text documents.\n\n2. **Goal:**\n   To automatically assign tags to text with high accuracy.\n\n3. **Core Concept:**\n   Supervised Learning on text data.\n\n4. **Technique / Method:**\n   1. Collect Dataset. 2. Preprocess (Clean/Stem). 3. Extract Features (TF-IDF). 4. Train Model (Naive Bayes/SVM). 5. Evaluate.\n\n5. **Applications:**\n   Detecting Hate Speech, Organizing Support Tickets, Filtering Spam Emails."
            }
        ]
    }
}

def populate_ml_unit4():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Unit 4: NLP"
        # Be careful to remove "Unit 4: Natural Language Processing"
        data = [u for u in data if "Natural Language Processing" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(ML_UNIT_4_DATA)
        print("Successfully replaced ML Unit 4 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_ml_unit4()
