from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import time
from nltk.stem import WordNetLemmatizer
import joblib
import scipy.sparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources if not already downloaded
def download_nltk_resources():
    try:
        # List of resources to download
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('vader_lexicon', 'sentiment/vader_lexicon')
        ]
        
        # Check and download each resource
        for resource, path in resources:
            try:
                nltk.data.find(path)
                print(f"Resource '{resource}' is already downloaded.")
            except LookupError:
                print(f"Downloading '{resource}'...")
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded '{resource}'.")
                
                # Special handling for tagger to verify it's properly installed
                if resource == 'averaged_perceptron_tagger':
                    try:
                        # Test if the tagger can be loaded
                        nltk.data.load('taggers/averaged_perceptron_tagger/english.pickle')
                        print("Successfully verified tagger resource.")
                    except LookupError:
                        print("Tagger resource path issue detected. Trying alternative download...")
                        # Try explicit download with full path
                        nltk.download('taggers/averaged_perceptron_tagger')
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        print("Will attempt to continue with available resources.")

# Initialize NLP components
download_nltk_resources()

# Initialize VADER sentiment analyzer
try:
    sid = SentimentIntensityAnalyzer()
    print("Successfully initialized VADER sentiment analyzer.")
except Exception as e:
    print(f"Error initializing VADER: {e}")
    sid = None
    print("VADER will not be available for sentiment analysis.")

# Initialize POS tagger with error handling
pos_tagger = None
try:
    # First try: Create a tagger that doesn't require pre-trained data
    try:
        from nltk.tag import PerceptronTagger
        # Initialize with load=False to avoid loading the pickle file
        pos_tagger = PerceptronTagger(load=False)
        print("Created PerceptronTagger without pre-trained data")
    except Exception as e1:
        print(f"Error creating minimal PerceptronTagger: {e1}")
        try:
            # Second try: Initialize NLTK's default pos_tag functionality
            # This is a wrapper around the correct tagger
            def nltk_wrapper_tagger(tokens):
                return nltk.pos_tag(tokens)
            
            # Test to make sure it works
            test_result = nltk_wrapper_tagger(['Test', 'sentence'])
            print(f"Created wrapper around nltk.pos_tag: {test_result}")
            pos_tagger = type('', (), {'tag': nltk_wrapper_tagger})()
        except Exception as e2:
            print(f"Error with nltk.pos_tag: {e2}")
            # Create a simple rule-based fallback tagger that doesn't rely on NLTK resources
            class SimpleNounTagger:
                def __init__(self):
                    self.non_nouns = {'the', 'and', 'but', 'for', 'with', 'a', 'an', 'at', 'by', 
                                    'from', 'in', 'to', 'is', 'are', 'was', 'were', 'be', 'been',
                                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                                    'can', 'could', 'should', 'very', 'really', 'not', 'no'}
                    self.verb_endings = {'ly', 'ing', 'ed', 'es', 'er', 'est', 'ize', 'ise'}
                    # Common nouns in reviews
                    self.common_nouns = {'room', 'hotel', 'staff', 'food', 'service', 'location', 
                                        'price', 'restaurant', 'breakfast', 'bed', 'bathroom',
                                        'shower', 'view', 'wifi', 'internet', 'cleanliness', 'place'}
                
                def tag(self, tokens):
                    return [(word, self._guess_pos(word)) for word in tokens]
                
                def _guess_pos(self, word):
                    word_lower = word.lower()
                    if word_lower in self.common_nouns:
                        return 'NN'  # Known nouns
                    if word_lower in self.non_nouns:
                        return 'DT' if word_lower in {'the', 'a', 'an'} else 'CC'
                    if any(word_lower.endswith(end) for end in self.verb_endings):
                        return 'VB'
                    if len(word_lower) > 3:
                        return 'NN'  # Assume longer words are nouns by default
                    return 'UNK'
            
            pos_tagger = SimpleNounTagger()
            print("Created simple rule-based noun tagger as fallback")

    # Test the tagger with a sample sentence to verify it works
    test_result = pos_tagger.tag(['This', 'is', 'a', 'test', 'sentence', 'with', 'nouns', 'room', 'food'])
    print(f"Tagger test result: {test_result}")
    
except Exception as e:
    print(f"All tagger initialization methods failed: {e}")
    print("Creating minimal fallback tagger")
    # Create an absolute minimal fallback that won't fail
    class MinimalTagger:
        def tag(self, tokens):
            # Just identify common nouns from our aspect list
            common_nouns = set(COMMON_ASPECTS)
            for aspect, variations in ASPECT_VARIATIONS.items():
                common_nouns.update(variations)
            
            return [(word, 'NN' if word.lower() in common_nouns or len(word) > 3 else 'UNK') 
                   for word in tokens]
    
    pos_tagger = MinimalTagger()

# Initialize lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    print("Successfully initialized WordNet lemmatizer.")
except Exception as e:
    print(f"Error initializing lemmatizer: {e}")
    lemmatizer = None
    print("Lemmatization will not be available.")

app = Flask(__name__)

# Enable CORS for frontend-backend communication
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Define common aspects in restaurant reviews
COMMON_ASPECTS = [
    # Restaurant aspects
    'food', 'service', 'price', 'ambiance', 'location', 'staff', 'menu', 'taste', 'quality', 'cleanliness', 
    'portion', 'value', 'atmosphere', 'decor', 'speed', 'presentation', 'reservation', 'parking', 'wait',
    # Hotel aspects 
    'room', 'bed', 'bathroom', 'shower', 'wifi', 'internet', 'breakfast', 'housekeeping', 'amenities', 
    'facilities', 'pool', 'gym', 'lobby', 'reception', 'view', 'noise', 'comfort', 'checkin', 'checkout'
]

# Additional word variations for stemming
ASPECT_VARIATIONS = {
    'staff': ['staffs', 'employee', 'employees', 'worker', 'workers', 'personnel', 'service', 'waiter', 'waitress'],
    'room': ['rooms', 'accommodation', 'suite', 'suites', 'bedroom', 'bedrooms'],
    'location': ['place', 'area', 'situated', 'situated', 'neighborhood', 'neighbourhood'],
    'food': ['meal', 'meals', 'breakfast', 'lunch', 'dinner', 'cuisine', 'dish', 'dishes'],
    'service': ['serving', 'assistance', 'help', 'attended', 'attending', 'staff', 'staffs'],
    'cleanliness': ['clean', 'dirty', 'filthy', 'neat', 'tidy', 'messy'],
    'price': ['cost', 'pricing', 'expensive', 'cheap', 'value', 'affordable', 'overpriced']
}

# Preprocess text by tokenizing, removing stopwords and lemmatizing
def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize (split by whitespace)
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back to string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        # Simple fallback preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

# Text Vectorizer
vectorizer = None

# Sentiment model
sentiment_model = None

# Train the model using train.csv
def train_sentiment_model():
    global vectorizer, sentiment_model
    
    model_path = './sentiment_model.pkl'
    vectorizer_path = './vectorizer.pkl'
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading pre-trained model and vectorizer...")
        try:
            sentiment_model = pickle.load(open(model_path, 'rb'))
            vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            return vectorizer, sentiment_model
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Will attempt to train a new model.")
            # Continue with training a new model

    print("Training new sentiment model using train.csv...")
    
    # Load and preprocess data
    try:
        train_df = pd.read_csv('train.csv')
        print(f"Loaded {len(train_df)} records from train.csv")
        
        # Drop unnecessary columns
        if 'User_ID' in train_df.columns:
            train_df = train_df.drop(['User_ID'], axis=1)
        if 'Browser_Used' in train_df.columns:
            train_df = train_df.drop(['Browser_Used'], axis=1)
        if 'Device_Used' in train_df.columns:
            train_df = train_df.drop(['Device_Used'], axis=1)
        
        # Map sentiment labels (happy: 1, not happy: 0)
        train_df['sentiment'] = train_df['Is_Response'].map({'happy': 1, 'not happy': 0})
        
        # Advanced text preprocessing
        print("Performing advanced text preprocessing...")
        
        # Create a more comprehensive preprocessor
        def advanced_preprocess(text):
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
            
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords but keep important sentiment words
            stop_words = set(stopwords.words('english'))
            sentiment_words = {'no', 'not', 'very', 'too', 'only', 'but', 'more', 'most', 'better', 'worse', 
                              'best', 'worst', 'really', 'pretty', 'quite', 'extremely'}
            filtered_stop_words = stop_words - sentiment_words
            tokens = [word for word in tokens if word not in filtered_stop_words]
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join tokens back to string
            preprocessed_text = ' '.join(tokens)
            return preprocessed_text
        
        # Apply advanced preprocessing
        train_df['processed_text'] = train_df['Description'].apply(advanced_preprocess)
        
        # Create additional features for better performance
        print("Creating additional features...")
        
        # Text length
        train_df['text_length'] = train_df['Description'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Count of sentiment words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'friendly', 'clean', 'recommend',
                         'best', 'perfect', 'wonderful', 'fantastic', 'favorite', 'love', 'enjoy', 'impressed']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'dirty', 'slow', 'rude', 'disappointing', 
                         'worst', 'horrible', 'mediocre', 'unfriendly', 'cold', 'expensive', 'avoid', 'never']
        
        # Count positive and negative words
        train_df['positive_count'] = train_df['Description'].apply(
            lambda x: sum(word in x.lower().split() for word in positive_words) if isinstance(x, str) else 0
        )
        train_df['negative_count'] = train_df['Description'].apply(
            lambda x: sum(word in x.lower().split() for word in negative_words) if isinstance(x, str) else 0
        )
        
        # Count exclamation and question marks
        train_df['exclamation_count'] = train_df['Description'].apply(
            lambda x: x.count('!') if isinstance(x, str) else 0
        )
        train_df['question_count'] = train_df['Description'].apply(
            lambda x: x.count('?') if isinstance(x, str) else 0
        )
        
        # Create sentiment score using VADER
        sid = SentimentIntensityAnalyzer()
        train_df['vader_score'] = train_df['Description'].apply(
            lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0
        )
        
        # Create feature vectors with advanced settings
        print("Creating feature vectors...")
        
        # Enhanced TF-IDF with n-grams and character n-grams
        vectorizer = TfidfVectorizer(
            max_features=10000,  # More features for better representation
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=3,           # Minimum document frequency
            max_df=0.9,         # Maximum document frequency
            sublinear_tf=True,  # Apply sublinear tf scaling (log scaling)
            use_idf=True,       # Use inverse document frequency
            analyzer='word',    # Analyze by words
            strip_accents='unicode',  # Remove accents
            token_pattern=r'\w{1,}'   # Include words of length 1 or more
        )
        
        # Create TF-IDF features
        X_text = vectorizer.fit_transform(train_df['processed_text'])
        
        # Get additional numerical features
        X_numerical = train_df[['text_length', 'positive_count', 'negative_count', 
                             'exclamation_count', 'question_count', 'vader_score']].values
        
        # Convert to sparse matrix for concatenation
        X_numerical_sparse = scipy.sparse.csr_matrix(X_numerical)
        
        # Combine TF-IDF and numerical features
        X = scipy.sparse.hstack([X_text, X_numerical_sparse])
        y = train_df['sentiment']
        
        # Split into train and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train an ensemble model for better performance
        print("Training ensemble model...")
        
        # Create a voting classifier with multiple models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        
        # Logistic Regression with class weight
        lr_model = LogisticRegression(
            max_iter=1000, 
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Create voting classifier
        sentiment_model = VotingClassifier(
            estimators=[
                ('lr', lr_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',  # Use predicted probabilities
            weights=[1, 1, 1]  # Equal weights
        )
        
        # Train the voting classifier
        sentiment_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = sentiment_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")
        
        # Save model and vectorizer
        print("Saving trained model and vectorizer...")
        try:
            pickle.dump(sentiment_model, open(model_path, 'wb'))
            pickle.dump(vectorizer, open(vectorizer_path, 'wb'))
            print("Model and vectorizer saved successfully.")
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")
        
        return vectorizer, sentiment_model
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        
        # Create and return fallback models
        try:
            # Create a simple TF-IDF vectorizer as fallback
            fallback_vectorizer = TfidfVectorizer(max_features=1000)
            # Train it on a small dummy dataset
            fallback_vectorizer.fit_transform(["positive review", "negative review"])
            
            # Create a simple LogisticRegression model as fallback
            fallback_model = LogisticRegression()
            # Create a dummy dataset and fit the model
            X_dummy = [[1], [0]]
            y_dummy = [1, 0]  # 1 for positive, 0 for negative
            fallback_model.fit(X_dummy, y_dummy)
            
            print("Created fallback models due to training error.")
            return fallback_vectorizer, fallback_model
        except Exception as fallback_error:
            print(f"Error creating fallback models: {fallback_error}")
            # Create absolutely minimal objects that can handle predict calls
            from sklearn.dummy import DummyClassifier
            
            dummy_model = DummyClassifier(strategy="most_frequent")
            dummy_model.fit([[0]], [1])  # Always predict positive
            
            class MinimalVectorizer:
                def transform(self, texts):
                    return scipy.sparse.csr_matrix([[0]])
                    
                def get_feature_names_out(self):
                    return np.array(["dummy"])
            
            dummy_vectorizer = MinimalVectorizer()
            
            print("Created minimal dummy models as last resort.")
            return dummy_vectorizer, dummy_model

# Ensure PunktSentenceTokenizer is properly initialized even without punkt_tab
try:
    from nltk.tokenize import PunktSentenceTokenizer
    punkt_tokenizer = PunktSentenceTokenizer()
except Exception as e:
    print(f"Could not initialize PunktSentenceTokenizer: {e}")
    punkt_tokenizer = None

# Sentence tokenization with improved fallback for punkt_tab errors
def tokenize_sentences(text):
    """Tokenize text into sentences with robust fallback mechanism"""
    if not text:
        return []
        
    # First attempt: Use NLTK's standard sentence tokenizer
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception as e:
        print(f"NLTK sent_tokenize failed: {e}")
        
    # Second attempt: Use initialized punkt_tokenizer
    if punkt_tokenizer:
        try:
            return punkt_tokenizer.tokenize(text)
        except Exception as e:
            print(f"PunktSentenceTokenizer failed: {e}")
    
    # Final fallback: Use regex-based tokenization
    print("Using fallback regex tokenization")
    return simple_sentence_tokenize(text)

# Simple sentence tokenizer as ultimate fallback
def simple_sentence_tokenize(text):
    """Simple sentence tokenization without NLTK dependencies"""
    if not text:
        return []
    # Split on sentence-ending punctuation with optional whitespace
    # This pattern covers most common sentence endings
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Handle case where no spaces after punctuation
    if len(sentences) <= 1:
        sentences = re.split(r'[.!?]+\s*', text)
    # Remove empty sentences and clean
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Aspect extraction model
def extract_aspects(text):
    """Extract aspects mentioned in the text"""
    if not text:
        return []
        
    try:
        # Convert to lowercase
        text_lower = text.lower()
        # Clean text of non-alphanumeric characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
        # Split by whitespace
        tokens = cleaned_text.split()
        
        # Look for aspect keywords directly
        aspects = []
        
        # First search for exact matches
        for aspect in COMMON_ASPECTS:
            if aspect in text_lower.split() or f" {aspect} " in f" {text_lower} ":
                aspects.append(aspect)
            # Check for multi-word aspects
            elif aspect in text_lower:
                aspects.append(aspect)
        
        # Then search for variations
        for aspect, variations in ASPECT_VARIATIONS.items():
            for variation in variations:
                if variation in text_lower.split() or f" {variation} " in f" {text_lower} ":
                    aspects.append(aspect)  # Add the canonical aspect name
                    break
        
        # Word-by-word similarity search for remaining tokens
        if len(aspects) < 1:  # If we don't have enough aspects, look deeper
            text_tokens = set(re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower).split())
            for token in text_tokens:
                if len(token) < 3:  # Skip very short tokens
                    continue
                for aspect in COMMON_ASPECTS:
                    # Stem/lemmatize comparison - check if token is part of aspect or vice versa
                    if token == aspect or (len(token) > 3 and (token in aspect or aspect in token)):
                        aspects.append(aspect)
        
        # Grammatical number normalization (e.g., "rooms" -> "room")
        normalized_aspects = []
        for aspect in aspects:
            # Check if this is a plural form by removing 's'
            if aspect.endswith('s') and aspect[:-1] in COMMON_ASPECTS:
                normalized_aspects.append(aspect[:-1])
            else:
                normalized_aspects.append(aspect)
                        
        # Return unique aspects
        return list(set(normalized_aspects))
    except Exception as e:
        print(f"Error in aspect extraction: {e}")
        # Simple fallback - just look for aspect keywords directly
        aspects = []
        for aspect in COMMON_ASPECTS:
            if aspect in text.lower():
                aspects.append(aspect)
        return list(set(aspects))

# Function to split review into aspect-specific segments
def split_review_by_aspects(review, aspects):
    """Split review text into segments related to each aspect with improved sentence-based approach."""
    print(f"Splitting review by aspects: '{review}'")
    
    aspect_segments = {}
    
    # If no aspects found, return empty dictionary
    if not aspects:
        return aspect_segments
    
    # First segment by periods, commas, 'and', 'but', etc.
    segments = []
    
    # Split the text into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', review)
    
    for sentence in sentences:
        # Further split on conjunctions and commas for more granular segmentation
        parts = re.split(r'\s+(?:and|but|however|although|yet|while)\s+|,\s+', sentence)
        segments.extend([part.strip() for part in parts if part.strip()])
    
    print(f"Split into segments: {segments}")
    
    # Initialize a mapping of each aspect to its potential segments
    aspect_segment_candidates = {aspect: [] for aspect in aspects}
    
    # For each segment, see which aspects it mentions
    for segment in segments:
        segment_lower = segment.lower()
        mentioned_aspects = []
        
        # Check for direct aspect mentions in this segment
        for aspect in aspects:
            aspect_lower = aspect.lower()
            # Check aspect and its variations
            if aspect_lower in segment_lower:
                mentioned_aspects.append(aspect)
                continue
                
            # Check for variations of this aspect
            if aspect in ASPECT_VARIATIONS:
                for variation in ASPECT_VARIATIONS[aspect]:
                    if variation.lower() in segment_lower:
                        mentioned_aspects.append(aspect)
                        break
        
        # Assign this segment to all mentioned aspects
        for aspect in mentioned_aspects:
            aspect_segment_candidates[aspect].append(segment)
    
    # For aspects without direct mentions, assign segments based on keyword relevance
    remaining_segments = segments.copy()
    for aspect, assigned_segments in aspect_segment_candidates.items():
        if assigned_segments:
            # Remove these segments from the remaining pool
            for segment in assigned_segments:
                if segment in remaining_segments:
                    remaining_segments.remove(segment)
    
    # For aspects without direct mentions, try to assign based on context
    for aspect in aspects:
        if not aspect_segment_candidates[aspect]:
            # Try to find most relevant segment from remaining
            if remaining_segments:
                # Simply assign the first unassigned segment if no direct match
                aspect_segment_candidates[aspect].append(remaining_segments[0])
                remaining_segments.pop(0)
            else:
                # If no segments remain, find the most general one
                aspect_segment_candidates[aspect].append(review)
    
    # Create the final aspect_segments dictionary
    for aspect, segments_list in aspect_segment_candidates.items():
        if segments_list:
            # Join all segments assigned to this aspect
            aspect_segments[aspect] = " ".join(segments_list)
        else:
            # Fallback to full review if no segments were assigned
            aspect_segments[aspect] = review
    
    # Handle special cases for common patterns
    for aspect in aspects:
        for segment in segments:
            segment_lower = segment.lower()
            # Explicit statements like "X is good" or "X is bad" should be prioritized
            for pattern in [f"{aspect.lower()} is good", f"{aspect.lower()} was good", 
                          f"{aspect.lower()} is bad", f"{aspect.lower()} was bad"]:
                if pattern in segment_lower:
                    aspect_segments[aspect] = segment
                    break
    
    # Final validation check for staffs/service consistency
    if "staff" in aspects and "service" in aspects:
        # If both staff and service are mentioned, make sure "staff" texts are correctly assigned
        for segment in segments:
            if "staff" in segment.lower():
                aspect_segments["staff"] = segment
            if "service" in segment.lower() and "staff" not in segment.lower():
                aspect_segments["service"] = segment
    
    # Print final segments for debugging
    for aspect, segment in aspect_segments.items():
        print(f"Final segment for {aspect}: '{segment}'")
    
    return aspect_segments

# Add common misspellings dictionary
MISSPELLING_MAP = {
    # Positive word misspellings
    'grate': 'great',
    'gud': 'good',
    'gr8': 'great',
    'excelent': 'excellent',
    'exellent': 'excellent',
    'awsome': 'awesome',
    'awsom': 'awesome',
    'fantastik': 'fantastic',
    'briliant': 'brilliant',
    'brillant': 'brilliant',
    'wonderfull': 'wonderful',
    'gd': 'good',
    'graet': 'great',
    'amzing': 'amazing',
    'perfct': 'perfect',
    'prefect': 'perfect',
    'nic': 'nice',
    'nyce': 'nice',
    
    # Negative word misspellings
    'terible': 'terrible',
    'terribl': 'terrible',
    'horribl': 'horrible',
    'horible': 'horrible',
    'badd': 'bad',
    'bd': 'bad',
    'poore': 'poor',
    'poorr': 'poor',
    'awfull': 'awful',
    'awfull': 'awful',
    'dissapointing': 'disappointing',
    'disapointing': 'disappointing'
}

def analyze_segment(segment, target_aspect=None):
    """Analyze a segment with simple rule-based approach first, then model"""
    # Check for simple positive/negative patterns first
    segment_lower = segment.lower()
    
    # Debug output
    print(f"Analyzing segment for {target_aspect}: {segment}")
    
    # Pre-process text to handle misspellings
    cleaned_words = []
    for word in segment_lower.split():
        # Check for misspellings and correct them
        if word in MISSPELLING_MAP:
            cleaned_words.append(MISSPELLING_MAP[word])
            print(f"Corrected misspelling: {word} -> {MISSPELLING_MAP[word]}")
        else:
            cleaned_words.append(word)
    
    # Reconstruct segment with corrected misspellings
    corrected_segment = " ".join(cleaned_words)
    if corrected_segment != segment_lower:
        print(f"Corrected text: {corrected_segment}")
        segment_lower = corrected_segment
    
    # Check for negative phrases that indicate issues for specific aspects
    if target_aspect == "ROOM" and any(phrase in segment_lower for phrase in ["smelled", "smell", "smelly", "funny smell", "odor", "stink"]):
        print(f"NEGATIVE match for {target_aspect} with smell-related terms")
        return {"label": "NEGATIVE", "score": 0.85}
    
    # 1. COMPLEX NEGATION DETECTION - check for various negation patterns
    # These are common ways people express negative sentiment
    complex_negation_patterns = [
        r'not (?:that|very|too|so) (?:much |very |too |really |extremely )?good',
        r'not (?:good|great|nice|wonderful|amazing|fantastic) (?:at all|enough)',
        r'could (?:be|have been) better',
        r'not (?:the best|impressive|outstanding)',
        r'didn\'t (?:like|enjoy|appreciate)',
        r'nothing (?:special|great|good)',
        r'barely (?:acceptable|adequate|passable)',
        r'far from (?:perfect|good|great|excellent)',
        r'leaves (?:much|a lot|something) to be desired',
        r'not (?:worth|up to) (?:the|it)',
        r'not as (?:good|great|nice) as'
    ]
    
    for pattern in complex_negation_patterns:
        if re.search(pattern, segment_lower, re.IGNORECASE):
            print(f"NEGATIVE match with complex negation pattern: {pattern}")
            return {"label": "NEGATIVE", "score": 0.9}
    
    # 2. DIRECT PATTERN MATCHING - highest priority for explicit statements
    # These patterns take absolute precedence over all other analysis
    direct_patterns = {
        "positive": [
            r'\b(?:is|was|are|were)\s+good\b',
            r'\bgood\b',
            r'\bgreat\b',
            r'\bexcellent\b',
            r'\bperfect\b',
            r'\bawesome\b',
            r'\bwonderful\b',
            r'\bamazing\b',
            r'\bfantastic\b'
        ],
        "negative": [
            r'\b(?:is|was|are|were)\s+(?:not|n\'t)\s+good\b',
            r'\b(?:is|was|are|were)\s+bad\b',
            r'\bbad\b',
            r'\bpoor\b',
            r'\bterrible\b',
            r'\bawful\b',
            r'\bhorrible\b',
            r'\bdisappointing\b'
        ]
    }
    
    # Check specific aspect patterns first with target aspect
    if target_aspect:
        target_lower = target_aspect.lower()
        # Direct patterns that explicitly mention the aspect
        # Format: "aspect is/was good/bad"
        explicit_aspect_positive = [
            f"{target_lower} is good",
            f"{target_lower} was good",
            f"{target_lower} is great", 
            f"{target_lower} was great",
            f"good {target_lower}",
            f"great {target_lower}",
            f"excellent {target_lower}"
        ]
        
        explicit_aspect_negative = [
            f"{target_lower} is not good",
            f"{target_lower} was not good", 
            f"{target_lower} is not that good",
            f"{target_lower} is not very good",
            f"{target_lower} is not much good",
            f"{target_lower} is bad",
            f"{target_lower} was bad",
            f"bad {target_lower}",
            f"poor {target_lower}",
            f"terrible {target_lower}"
        ]
        
        # Special case for "staff/staffs was good" - force positive
        if target_lower in ["staff", "service"] and ("staff was good" in segment_lower or "staffs was good" in segment_lower):
            print(f"SPECIAL POSITIVE match for {target_aspect} with 'was good' pattern")
            return {"label": "POSITIVE", "score": 0.95}
            
        # Check explicit negative patterns first
        for pattern in explicit_aspect_negative:
            if pattern in segment_lower:
                print(f"EXPLICIT NEGATIVE match for {target_aspect} with pattern: {pattern}")
                return {"label": "NEGATIVE", "score": 0.95}
        
        # Then check explicit positive patterns
        for pattern in explicit_aspect_positive:
            if pattern in segment_lower:
                print(f"EXPLICIT POSITIVE match for {target_aspect} with pattern: {pattern}")
                return {"label": "POSITIVE", "score": 0.95}
    
    # 3. GENERAL PATTERN MATCHING for the entire segment
    # Check for negative patterns (negative takes precedence)
    for pattern in direct_patterns["negative"]:
        if re.search(pattern, segment_lower, re.IGNORECASE):
            print(f"GENERAL NEGATIVE match with pattern: {pattern}")
            return {"label": "NEGATIVE", "score": 0.9}
            
    # Check for positive patterns
    for pattern in direct_patterns["positive"]:
        if re.search(pattern, segment_lower, re.IGNORECASE):
            print(f"GENERAL POSITIVE match with pattern: {pattern}")
            return {"label": "POSITIVE", "score": 0.9}
    
    # 4. COMPREHENSIVE NEGATION HANDLING
    negation_words = ['not', 'never', "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                     "hardly", "barely", "scarcely", "rarely", "seldom", "nor"]
    negation_present = False
    negation_index = -1
    
    # Find position of negation words
    words = segment_lower.split()
    for i, word in enumerate(words):
        if word in negation_words:
            negation_present = True
            negation_index = i
            break
    
    # 5. KEYWORD COUNTING for general sentiment
    positive_words = [
        'good', 'great', 'excellent', 'fantastic', 'nice', 'amazing', 'wonderful', 
        'awesome', 'perfect', 'clean', 'comfortable', 'spacious', 'convenient',
        'friendly', 'helpful', 'professional', 'enjoyed', 'recommend', 'recommended',
        'enjoyable', 'satisfied', 'satisfying', 'love', 'loved', 'lovely', 'beautiful', 'fresh'
    ]
    
    negative_words = [
        'bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointing', 'not good',
        'dirty', 'small', 'tiny', 'cramped', 'uncomfortable', 'inconvenient',
        'rude', 'unhelpful', 'unfriendly', 'unprofessional', 'dissatisfied',
        'broken', 'damaged', 'noisy', 'smelly', 'smelled', 'unpleasant', 'mediocre',
        'lacking', 'lacks', 'lacked', 'issue', 'issues', 'problem', 'problems', 'funny'
    ]
    
    # Adjust for negation flipping sentiment of nearby positive words
    if negation_present and negation_index >= 0:
        # Check if the negation applies to positive words within 4 words
        # For example: "not very good" or "not that great at all"
        window_end = min(len(words), negation_index + 5)
        window = words[negation_index:window_end]
        
        for pos_word in positive_words:
            if pos_word in window:
                print(f"NEGATIVE match based on negated positive word: {pos_word}")
                return {"label": "NEGATIVE", "score": 0.85}
    
    # Count positive and negative words
    pos_count = sum(1 for word in positive_words if word in segment_lower.split())
    neg_count = sum(1 for word in negative_words if word in segment_lower.split())
    
    # Final sentiment based on word count
    if pos_count > neg_count + 1:
        print(f"POSITIVE match based on keyword count: {pos_count} positive vs {neg_count} negative")
        return {"label": "POSITIVE", "score": 0.8}
    elif neg_count > pos_count + 1:
        print(f"NEGATIVE match based on keyword count: {neg_count} negative vs {pos_count} positive")
        return {"label": "NEGATIVE", "score": 0.8}
    elif pos_count == neg_count and pos_count > 0:
        print(f"NEUTRAL match with mixed sentiment")
        return {"label": "NEUTRAL", "score": 0.7}
    
    # 6. FALLBACK TO ML MODEL
    # If no pattern matches, use the ML model
    ml_result = predict_sentiment(segment)
    print(f"ML model result: {ml_result}")
    return {"label": ml_result["label"], "score": normalize_confidence(ml_result["score"])}

def analyze_text(segment, target_aspect=None):
    """Enhanced universal sentiment analysis for any review text."""
    # First try the rule-based approach with enhanced pattern matching
    rule_result = analyze_segment(segment, target_aspect)
    
    # If rule-based approach is highly confident (>0.85), use it
    if rule_result["score"] > 0.85:
        return {"label": rule_result["label"], "score": normalize_confidence(rule_result["score"])}
    
    # Otherwise, use the ensemble approach combining rules, ML and VADER
    try:
        # 1. Try VADER sentiment analysis
        vader_result = analyze_sentiment_nlp(segment, target_aspect)
        
        # 2. Get ML model prediction
        ml_result = predict_sentiment(segment)
        
        # 3. Combine results based on confidence
        # If rule and VADER agree, use rule with higher confidence
        if rule_result["label"] == vader_result["label"]:
            combined_score = normalize_confidence(max(rule_result["score"], vader_result["score"]))
            return {"label": rule_result["label"], "score": combined_score}
        
        # If VADER is highly confident, use it
        if vader_result["score"] > 0.8:
            return {"label": vader_result["label"], "score": normalize_confidence(vader_result["score"])}
        
        # If ML and VADER agree, combine them
        if ml_result["label"] == vader_result["label"]:
            combined_score = normalize_confidence((ml_result["score"] + vader_result["score"]) / 2)
            return {"label": ml_result["label"], "score": combined_score}
        
        # If nothing agrees strongly, use the most confident result
        scores = [
            (rule_result["label"], normalize_confidence(rule_result["score"])),
            (vader_result["label"], normalize_confidence(vader_result["score"])),
            (ml_result["label"], normalize_confidence(ml_result["score"]))
        ]
        
        # Sort by confidence score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most confident result
        return {"label": scores[0][0], "score": scores[0][1]}
    
    except Exception as e:
        print(f"Error in ensemble analysis: {e}")
        # Fall back to rule-based with normalized confidence
        return {"label": rule_result["label"], "score": normalize_confidence(rule_result["score"])}

# Define aspect-specific patterns for sentiment detection
POSITIVE_PATTERNS = {
    "room": [
        r'room\s+(?:is|was)\s+(?:good|great|nice|clean|excellent|comfortable)',
        r'good\s+room', r'nice\s+room', r'great\s+room', 
        r'comfortable\s+room', r'clean\s+room', r'spacious\s+room'
    ],
    "location": [
        r'location\s+(?:is|was)\s+(?:good|great|nice|excellent|perfect|convenient)',
        r'good\s+location', r'great\s+location', r'nice\s+location', 
        r'excellent\s+location', r'perfect\s+location', r'convenient\s+location'
    ],
    "staff": [
        r'staff\s+(?:is|was|are|were)\s+(?:good|friendly|helpful|nice|professional)',
        r'good\s+staff', r'friendly\s+staff', r'helpful\s+staff', 
        r'nice\s+staff', r'professional\s+staff'
    ],
    "cleanliness": [
        r'(?:is|was|are|were)\s+clean', r'clean\s+room', r'clean\s+hotel', 
        r'cleanliness\s+(?:is|was)\s+good', r'good\s+cleanliness'
    ],
    "food": [
        r'food\s+(?:is|was)\s+(?:good|great|excellent|delicious|tasty)',
        r'good\s+food', r'great\s+food', r'excellent\s+food', 
        r'delicious\s+food', r'tasty\s+food'
    ],
    "service": [
        r'service\s+(?:is|was)\s+(?:good|great|excellent|fast|friendly)',
        r'good\s+service', r'great\s+service', r'excellent\s+service',
        r'friendly\s+service', r'fast\s+service'
    ]
}

NEGATIVE_PATTERNS = {
    "room": [
        r'room\s+(?:is|was)\s+(?:bad|poor|dirty|small|tiny|not\s+good)',
        r'bad\s+room', r'dirty\s+room', r'small\s+room', r'tiny\s+room'
    ],
    "location": [
        r'location\s+(?:is|was)\s+(?:bad|poor|terrible|awful|not\s+good|too\s+bad)',
        r'bad\s+location', r'poor\s+location', r'terrible\s+location',
        r'awful\s+location', r'inconvenient\s+location'
    ],
    "staff": [
        r'staff\s+(?:is|was|are|were)\s+(?:bad|rude|unhelpful|unfriendly|not\s+good)',
        r'bad\s+staff', r'rude\s+staff', r'unhelpful\s+staff', 
        r'unfriendly\s+staff', r'unprofessional\s+staff',
        r'staff\s+(?:also|too)\s+not\s+good', r'staff\s+not\s+good'
    ],
    "cleanliness": [
        r'(?:is|was|are|were)\s+dirty', r'dirty\s+room', r'dirty\s+hotel',
        r'cleanliness\s+(?:is|was)\s+poor', r'poor\s+cleanliness'
    ],
    "food": [
        r'food\s+(?:is|was)\s+(?:bad|poor|terrible|awful|not\s+good|tasteless)',
        r'bad\s+food', r'poor\s+food', r'terrible\s+food',
        r'awful\s+food', r'tasteless\s+food'
    ],
    "service": [
        r'service\s+(?:is|was)\s+(?:bad|poor|terrible|slow|rude|not\s+good)',
        r'bad\s+service', r'poor\s+service', r'terrible\s+service',
        r'slow\s+service', r'rude\s+service'
    ]
}

# Add NLP-based sentiment analysis
def analyze_sentiment_nlp(text, target_aspect=None):
    """Use NLP techniques including VADER to analyze sentiment"""
    try:
        # First try with VADER if available
        if sid:
            # Get VADER sentiment scores
            vader_scores = sid.polarity_scores(text)
            
            # Check for strong negative or positive sentiments
            compound_score = vader_scores['compound']
            
            # Check aspect-specific patterns first (these override VADER)
            if target_aspect:
                # Check for negative patterns for this aspect
                if target_aspect in NEGATIVE_PATTERNS:
                    for pattern in NEGATIVE_PATTERNS[target_aspect]:
                        if re.search(pattern, text, re.IGNORECASE):
                            return {"label": "NEGATIVE", "score": 0.9}
                
                # Check for positive patterns for this aspect
                if target_aspect in POSITIVE_PATTERNS:
                    for pattern in POSITIVE_PATTERNS[target_aspect]:
                        if re.search(pattern, text, re.IGNORECASE):
                            return {"label": "POSITIVE", "score": 0.9}
            
            # Then use VADER's compound score
            if compound_score <= -0.2:  # More lenient threshold for negative
                return {"label": "NEGATIVE", "score": normalize_confidence(min(0.9, abs(compound_score) * 1.5))}
            elif compound_score >= 0.2:  # Standard threshold for positive
                return {"label": "POSITIVE", "score": normalize_confidence(min(0.9, abs(compound_score) * 1.5))}
                
            # For borderline cases, check negation words
            negations = ["not", "no", "never", "neither", "nor", "barely", "hardly", "doesn't", "isn't", "wasn't", 
                         "weren't", "haven't", "hasn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "don't"]
            for neg in negations:
                if f" {neg} " in f" {text.lower()} ":
                    return {"label": "NEGATIVE", "score": 0.75}  # Presence of negation often indicates negative sentiment
        
        # Fallback to original approach using aspect-specific patterns
        result = analyze_segment(text, target_aspect)
        return {"label": result["label"], "score": normalize_confidence(result["score"])}
            
    except Exception as e:
        print(f"Error in NLP sentiment analysis: {e}")
        # Fallback to original method
        result = analyze_segment(text, target_aspect)
        return {"label": result["label"], "score": normalize_confidence(result["score"])}

# Function to predict sentiment using trained model
def predict_sentiment(text):
    """Predict sentiment using the trained model"""
    try:
        if sentiment_model is None or vectorizer is None:
            raise ValueError("Model not trained yet")
        
        # Apply the same preprocessing as in training
        def advanced_preprocess(text):
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
            
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords but keep important sentiment words
            stop_words = set(stopwords.words('english'))
            sentiment_words = {'no', 'not', 'very', 'too', 'only', 'but', 'more', 'most', 'better', 'worse', 
                              'best', 'worst', 'really', 'pretty', 'quite', 'extremely'}
            filtered_stop_words = stop_words - sentiment_words
            tokens = [word for word in tokens if word not in filtered_stop_words]
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join tokens back to string
            preprocessed_text = ' '.join(tokens)
            return preprocessed_text
        
        # Preprocess the text
        preprocessed = advanced_preprocess(text)
        
        # Create the same additional features
        positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'friendly', 'clean', 'recommend',
                         'best', 'perfect', 'wonderful', 'fantastic', 'favorite', 'love', 'enjoy', 'impressed']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'dirty', 'slow', 'rude', 'disappointing', 
                         'worst', 'horrible', 'mediocre', 'unfriendly', 'cold', 'expensive', 'avoid', 'never']
        
        # Calculate numerical features
        text_length = len(text)
        positive_count = sum(word in text.lower().split() for word in positive_words)
        negative_count = sum(word in text.lower().split() for word in negative_words)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Get VADER score
        sid = SentimentIntensityAnalyzer()
        vader_score = sid.polarity_scores(text)['compound']
        
        # Vectorize the text
        text_vector = vectorizer.transform([preprocessed])
        
        # Create additional features matrix
        additional_features = np.array([[
            text_length, positive_count, negative_count, 
            exclamation_count, question_count, vader_score
        ]])
        
        # Convert to sparse matrix for concatenation
        additional_features_sparse = scipy.sparse.csr_matrix(additional_features)
        
        # Combine features
        combined_vector = scipy.sparse.hstack([text_vector, additional_features_sparse])
        
        # Predict probability
        proba = sentiment_model.predict_proba(combined_vector)[0]
        
        # Define a neutral range around the middle (0.5)
        neutral_threshold = 0.15  # If within +/- this value of 0.5, consider neutral
        
        if proba[1] > 0.5 + neutral_threshold:  # Clearly positive
            sentiment = "POSITIVE"
            confidence = proba[1]
        elif proba[1] < 0.5 - neutral_threshold:  # Clearly negative
            sentiment = "NEGATIVE"
            confidence = proba[0]
        else:  # In the neutral zone
            sentiment = "NEUTRAL"
            # Calculate how close to perfect neutrality (0.5)
            confidence = 1.0 - (abs(0.5 - proba[1]) / neutral_threshold)
        
        return {
            "label": sentiment,
            "score": float(confidence)
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Use the VADER analyzer as fallback
        try:
            sid = SentimentIntensityAnalyzer()
            scores = sid.polarity_scores(text)
            
            if scores['compound'] >= 0.05:
                return {"label": "POSITIVE", "score": min(0.9, abs(scores['compound']) * 1.5)}
            elif scores['compound'] <= -0.05:
                return {"label": "NEGATIVE", "score": min(0.9, abs(scores['compound']) * 1.5)}
            else:
                return {"label": "NEUTRAL", "score": 0.7}
        except:
            # Fallback - simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'friendly', 'clean', 'recommend']
            negative_words = ['bad', 'poor', 'terrible', 'awful', 'dirty', 'slow', 'rude', 'disappointing']
            neutral_words = ['okay', 'ok', 'average', 'decent', 'fair', 'moderate', 'standard', 'acceptable']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            neutral_count = sum(1 for word in neutral_words if word in text_lower)
            
            # Determine sentiment based on keyword counts
            if positive_count > negative_count and positive_count > neutral_count:
                return {"label": "POSITIVE", "score": 0.7}
            elif negative_count > positive_count and negative_count > neutral_count:
                return {"label": "NEGATIVE", "score": 0.7}
            else:
                return {"label": "NEUTRAL", "score": 0.7}

# Add the NLP-based aspect extraction function
def extract_aspects_nlp(text):
    """Extract aspects using NLP techniques including POS tagging"""
    # First get aspects using the basic approach
    basic_aspects = extract_aspects(text)
    
    try:
        # Tokenize the text
        try:
            tokens = nltk.word_tokenize(text.lower())
            print("Successfully tokenized text")
        except Exception as e_token:
            print(f"Error in tokenization: {e_token}")
            # Fallback tokenization - simple split by whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
            print("Using regex-based tokenization as fallback")
        
        # Perform part-of-speech tagging with improved error handling
        nouns = []
        tagging_success = False
        
        # Method 1: Use our custom tagger that should work in all cases
        if pos_tagger:
            try:
                tagged = pos_tagger.tag(tokens)
                # Extract nouns (look for NN tags)
                nouns = [word for word, pos in tagged if pos.startswith('NN')]
                print(f"Successfully tagged text using custom tagger, found {len(nouns)} nouns")
                tagging_success = True
            except Exception as e:
                print(f"Custom tagger failed: {e}")
        
        # Method 2: Direct word filtering if no nouns found
        if not tagging_success or not nouns:
            print("Using direct word filtering for noun extraction")
            # Skip common non-nouns and words with typical verb/adjective/adverb endings
            common_non_nouns = {'the', 'and', 'but', 'for', 'with', 'a', 'an', 'at', 'by', 
                               'from', 'in', 'to', 'is', 'are', 'was', 'were', 'be', 'been',
                               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                               'can', 'could', 'should', 'very', 'really', 'not', 'no'}
            verb_endings = {'ly', 'ing', 'ed', 'es', 's', 'er', 'est', 'ize', 'ise'}
            
            for token in tokens:
                # Skip very short tokens and common non-nouns
                if len(token) < 3 or token in common_non_nouns:
                    continue
                
                # Simple heuristic: check if the token ends with common verb/adjective/adverb endings
                if not any(token.endswith(ending) for ending in verb_endings):
                    nouns.append(token)
            
            print(f"Direct word filtering found {len(nouns)} potential nouns")
        
        # Method 3: Context-based noun extraction - always run this to improve results
        print("Performing context-based noun extraction")
        # Check words near common aspect keywords
        aspect_sentences = []
        
        # Find sentences containing aspects
        sentences = re.split(r'[.!?]+', text.lower())
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for common aspects in this sentence
            for aspect in COMMON_ASPECTS:
                if aspect in sentence:
                    aspect_sentences.append(sentence)
                    break
                    
            # Also check aspect variations
            for aspect, variations in ASPECT_VARIATIONS.items():
                if any(variation in sentence for variation in variations):
                    aspect_sentences.append(sentence)
                    break
        
        # Extract candidate nouns from sentences containing aspects
        for sentence in aspect_sentences:
            sentence_tokens = re.findall(r'\b\w+\b', sentence)
            for token in sentence_tokens:
                if (len(token) > 3 and 
                    token not in common_non_nouns and 
                    not any(token.endswith(ending) for ending in verb_endings) and
                    token not in nouns):
                    nouns.append(token)
        
        # Find aspects in our defined list
        nlp_aspects = []
        
        # First check direct matches with common aspects
        for noun in nouns:
            # Check if noun directly matches a common aspect
            if noun in COMMON_ASPECTS:
                nlp_aspects.append(noun)
                continue
                
            # Check if noun matches aspect variations
            matched = False
            for aspect, variations in ASPECT_VARIATIONS.items():
                if noun in variations:
                    nlp_aspects.append(aspect)
                    matched = True
                    break
            
            # Check for partial matches if no direct match found
            if not matched and len(noun) > 3:
                for aspect in COMMON_ASPECTS:
                    # Check if noun contains aspect or aspect contains noun
                    if noun in aspect or aspect in noun:
                        nlp_aspects.append(aspect)
                        break
        
        # Method 4: Direct aspect keyword search - always do this for completeness
        print("Performing direct aspect keyword search")
        text_lower = " " + text.lower() + " "  # Add spaces to ensure word boundaries
        
        for aspect in COMMON_ASPECTS:
            # Check for exact word
            if (f" {aspect} " in text_lower or 
                f" {aspect}." in text_lower or 
                f" {aspect}," in text_lower or
                f" {aspect}!" in text_lower or
                f" {aspect}?" in text_lower):
                nlp_aspects.append(aspect)
                continue
            
            # Check variations
            if aspect in ASPECT_VARIATIONS:
                for variation in ASPECT_VARIATIONS[aspect]:
                    if (f" {variation} " in text_lower or 
                        f" {variation}." in text_lower or 
                        f" {variation}," in text_lower or
                        f" {variation}!" in text_lower or
                        f" {variation}?" in text_lower):
                        nlp_aspects.append(aspect)
                        break
        
        # Combine both approaches and remove duplicates
        combined_aspects = list(set(basic_aspects + nlp_aspects))
        
        # If still no aspects, add a default general aspect for sentiment analysis
        if not combined_aspects:
            print("No aspects found, using default 'general' aspect")
            combined_aspects = ["general"]
        else:
            print(f"Found aspects: {combined_aspects}")
            
        return combined_aspects
    except Exception as e:
        print(f"Error in NLP aspect extraction: {e}")
        # Fall back to basic extraction, or at minimum return a default aspect
        if basic_aspects:
            return basic_aspects
        return ["general"]

# Function to ensure confidence values are in the valid range
def normalize_confidence(confidence):
    """Normalize confidence to be between 0 and 1"""
    # Ensure confidence is a float
    try:
        value = float(confidence)
    except (ValueError, TypeError):
        print(f"Warning: Invalid confidence value {confidence}, defaulting to 0.5")
        return 0.5
        
    # Check if already in range 0-1
    if 0 <= value <= 1:
        return value
        
    # Check if it's a percentage (0-100)
    if 1 < value <= 100:
        return value / 100
        
    # For any other cases, cap at 0-1 range
    return max(0, min(1, value))

@app.route('/analyze', methods=['POST'])
def analyze_review():
    """Analyze a review for overall and aspect-based sentiment."""
    start_time = time.time()
    
    # Check if models are loaded
    if sentiment_model is None:
        return jsonify({
            'error': 'Model is not trained yet. Please wait for training to complete.',
            'overall': 'UNKNOWN',
            'confidence': 0,
            'aspects': []
        }), 503
    
    data = request.get_json()
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'No review text provided'}), 400
    
    try:
        # Extract aspects from the review using NLP
        aspects = extract_aspects_nlp(review)
        
        # Direct sentiment analysis on the whole text
        direct_overall_result = analyze_text(review)
        print(f"DEBUG - Direct overall result: {direct_overall_result}")
        
        # Perform aspect-based sentiment analysis
        aspect_results = []
        aspect_sentiments = []
        
        if aspects:
            # Split the review text by aspects
            aspect_segments = split_review_by_aspects(review, aspects)
            
            # Analyze sentiment for each aspect
            for aspect, text in aspect_segments.items():
                try:
                    # Use enhanced aspect-specific sentiment analysis
                    aspect_result = analyze_text(text, target_aspect=aspect)
                    print(f"DEBUG - {aspect} raw result: {aspect_result}")
                    
                    aspect_sentiment = aspect_result["label"]
                    # Ensure confidence is between 0 and 1
                    aspect_confidence = normalize_confidence(aspect_result["score"])
                    print(f"DEBUG - {aspect} normalized confidence: {aspect_confidence}")
                    
                    # Store the aspect sentiment for later aggregation
                    aspect_sentiments.append((aspect_sentiment, aspect_confidence, aspect))
                except Exception as e:
                    print(f"Error analyzing aspect {aspect}: {str(e)}")
                    # Fallback for very short text that might not vectorize well
                    aspect_sentiment = direct_overall_result["label"]
                    aspect_confidence = 0.6
                    aspect_sentiments.append((aspect_sentiment, aspect_confidence, aspect))
                
                # Format confidence as percentage for display
                display_confidence = round(aspect_confidence * 100, 2)
                print(f"DEBUG - {aspect} display confidence: {display_confidence}%")
                
                aspect_results.append({
                    'aspect': aspect,
                    'sentiment': aspect_sentiment,
                    'confidence': display_confidence,
                    'text': text
                })
        
        # Determine overall sentiment based on aspect sentiments and direct analysis
        if aspect_sentiments:
            # Convert sentiment labels to numerical values for calculation
            sentiment_values = {
                "POSITIVE": 1,
                "NEUTRAL": 0,
                "NEGATIVE": -1
            }
            
            # Calculate weighted average of aspect sentiments
            total_weight = 0
            weighted_sum = 0
            
            # Sort by confidence - higher confidence aspects have more impact
            aspect_sentiments.sort(key=lambda x: x[1], reverse=True)
            
            # Priority aspects with common weightings
            aspect_priorities = {
                'food': 1.5,       # Food is usually a primary concern for restaurants
                'service': 1.3,    # Service is also very important
                'staff': 1.3,      # Staff experience matters a lot
                'cleanliness': 1.2,# Cleanliness is important
                'room': 1.5,       # For hotels, rooms are critical
                'location': 1.2,   # Location matters but not as much as core experience
                'value': 1.2,      # Value for money is important
                'price': 1.2,      # Price concerns matter
                'general': 1.0     # Default weight
            }
            
            # Apply weighted sentiment calculation
            for sentiment, confidence, aspect in aspect_sentiments:
                # Get aspect priority weight (default 1.0 if not in dictionary)
                aspect_weight = aspect_priorities.get(aspect.lower(), 1.0)
                
                # Calculate effective weight (confidence × priority)
                effective_weight = confidence * aspect_weight
                
                weighted_sum += sentiment_values[sentiment] * effective_weight
                total_weight += effective_weight
            
            if total_weight > 0:
                aspect_avg = weighted_sum / total_weight
                print(f"DEBUG - Aspect average score: {aspect_avg}")
                
                # Convert back to categorical sentiment with more nuanced thresholds
                if aspect_avg > 0.25:
                    overall_sentiment = "POSITIVE"
                    overall_confidence = min(0.95, 0.6 + abs(aspect_avg) * 0.35)
                elif aspect_avg < -0.25:
                    overall_sentiment = "NEGATIVE"
                    overall_confidence = min(0.95, 0.6 + abs(aspect_avg) * 0.35)
                else:
                    overall_sentiment = "NEUTRAL"
                    overall_confidence = 1.0 - (abs(aspect_avg) / 0.25) * 0.3
                
                print(f"DEBUG - Initial overall confidence: {overall_confidence}")
                
                # Ensure within 0-1 range
                overall_confidence = normalize_confidence(overall_confidence)
                print(f"DEBUG - Normalized overall confidence: {overall_confidence}")
                
                # Consistency check: If strong negative aspects, overall should reflect that
                if any(s[0] == "NEGATIVE" and s[1] > 0.85 for s in aspect_sentiments[:2]):
                    # If any of the top 2 high-confidence aspects are negative, ensure overall isn't too positive
                    if overall_sentiment == "POSITIVE":
                        overall_sentiment = "NEUTRAL"
                        overall_confidence = 0.7
                
                # Consistency check: If all positive aspects, overall shouldn't be neutral/negative
                if overall_sentiment != "POSITIVE" and all(s[0] == "POSITIVE" for s in aspect_sentiments):
                    overall_sentiment = "POSITIVE"
                    overall_confidence = 0.75
            else:
                # Fallback to direct analysis if weights sum to zero
                overall_sentiment = direct_overall_result["label"]
                overall_confidence = normalize_confidence(direct_overall_result["score"])
        else:
            # No aspect analysis, use direct analysis
            overall_sentiment = direct_overall_result["label"]
            overall_confidence = normalize_confidence(direct_overall_result["score"])
        
        # Format confidence as percentage for display (0-100 scale with 2 decimal places)
        display_confidence = round(overall_confidence * 100, 2)
        print(f"DEBUG - Final overall confidence: {overall_confidence}, display: {display_confidence}%")
        
        # Calculate processing time
        processing_time = time.time() - start_time

        response_data = {
            'overall': overall_sentiment,
            'confidence': display_confidence,
            'aspects': aspect_results,
            'processing_time': processing_time
        }
        
        print(f"DEBUG - Final JSON response confidence value: {response_data['confidence']}")
        return jsonify(response_data)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return error response if something goes wrong
        return jsonify({
            'error': f"Analysis failed: {str(e)}",
            'overall': "UNKNOWN",
            'confidence': 0,
            'aspects': [],
            'processing_time': time.time() - start_time
        })

if __name__ == '__main__':
    try:
        print("Training sentiment model...")
        train_sentiment_model()
        print("Starting Flask server...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Server crashed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to close...")