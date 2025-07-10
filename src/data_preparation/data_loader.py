from unittest import case

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# pd.set_option('display.max_columns', None)

# --- NLTK Data Downloads (Run this ONCE, then comment out or remove) ---
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_en')

# ---------------------------------------------------------------------

NEGATING_WORDS = {'not', 'no', 'n\'t', 'never', 'none', 'neither', 'nor', 'hardly', 'barely', 'scarcely', 'seldom'}
stop_words = set(stopwords.words('english'))
wnl = nltk.stem.WordNetLemmatizer()
count_vectorizer = CountVectorizer()
logistic_regression = LogisticRegression(max_iter=10000)

test_size = 0.2
random_state = 42


def map_penn_to_wordnet(tag: str) -> str:
    # Handle empty tag case defensively
    if not tag:
        return 'n'

    # Map Penn Treebank tags to WordNet tags
    # WordNet accepts 'n', 'v', 'a', 'r' (noun, verb, adjective, adverb)
    # We'll map based on the general category signified by the first letter
    # and provide a default if no specific mapping is found.
    tag_map = {
        'J': 'a',  # Adjective (e.g., JJ, JJR, JJS)
        'V': 'v',  # Verb (e.g., VB, VBD, VBG, VBN, VBP, VBZ)
        'N': 'n',  # Noun (e.g., NN, NNS, NNP, NNPS)
        'R': 'r'  # Adverb (e.g., RB, RBR, RBS)
    }

    # Get the first letter of the tag
    first_letter_tag = tag[0]

    # Use .get() with a default value of 'n' (noun) if no mapping is found
    # This aligns with WordNetLemmatizer's default behavior
    return tag_map.get(first_letter_tag, 'n')


def handle_negation(token_lst: list) -> list:
    i = 0
    processed_tokens = []
    while i < len(token_lst):
        token = token_lst[i]
        if token_lst[i] in NEGATING_WORDS:
            if i + 1 < len(token_lst):
                processed_tokens.append(f'{token_lst[i + 1]}_NEG')
                i += 2
            else:
                processed_tokens.append(token)
                i += 1

        else:
            processed_tokens.append(token)
            i += 1
    return processed_tokens


def encode_sentiment(sentiment: str) -> int:
    match sentiment:
        case 'positive':
            return 1
        case 'negative':
            return -1
    return 0

sample_feedback_data: dict = {'feedback_id': [1, 2, 3, 4],
                              'ai_tool_name': ['ChatGPT 4.0', 'Midjourney', 'Copilot', 'Internal Tool X'],
                              'user_id': ['user_A', 'user_B', 'user_C', 'user_D'],
                              'timestamp': ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04'],
                              'feedback_text': [
                                  'The interface is super intuitive, but it crashes sometimes during heavy use',
                                  'Image generation is amazing, but the commands are confusing. I wish it was more '
                                  'stable.',
                                  'Great for code suggestions, really speeds up my work. Never had an issue with '
                                  'stability.',
                                  'Hard to navigate, very slow, and keeps freezing. Not worth the effort.'],
                              'sentiment': ['neutral', 'neutral', 'positive', 'negative']}

df = pd.DataFrame.from_records(sample_feedback_data)

df['sentiment_numeric'] = df['sentiment'].apply(encode_sentiment)

df['tokenized_feedback'] = df['feedback_text'].apply(word_tokenize)

df['lowercased_tokens'] = df['tokenized_feedback'].apply(lambda x: [i.lower() for i in x])

df['filtered_tokens'] = df['lowercased_tokens'].apply(lambda x: [i for i in x if i not in stop_words])

df['filtered_tokens'] = df['filtered_tokens'].apply(
    lambda x: [i for i in x if i.isalpha() or i == 'n\'t'])

df['pos_tagged_tokens'] = df['filtered_tokens'].apply(pos_tag)

df['lemmatized_tokens'] = df['pos_tagged_tokens'].apply(
    lambda x: handle_negation([wnl.lemmatize(i[0], map_penn_to_wordnet(i[1])) for i in x]))

joined_tokens = df['lemmatized_tokens'].str.join(' ')
bow_features = count_vectorizer.fit_transform(joined_tokens)

X = bow_features
y = df['sentiment_numeric'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

joblib.dump(count_vectorizer, '../../trained_models/count_vectorizer.joblib')
joblib.dump(logistic_regression, '../../trained_models/sentiment_model.joblib')


new_sample_txt = 'I hate how slow this takes to load. My workflow is constantly interrupted.'
loaded_vectorizer = joblib.load('../../trained_models/count_vectorizer.joblib')
loaded_model = joblib.load('../../trained_models/sentiment_model.joblib')

tokenized_new_sample = word_tokenize(new_sample_txt)
lowercase_new_sample = [i.lower() for i in tokenized_new_sample]
filtered_new_sample = [i for i in lowercase_new_sample if i not in stop_words]
filtered_new_sample = [i for i in filtered_new_sample if i.isalpha() or i == 'n\'t']
pos_tagged_new_sample = pos_tag(filtered_new_sample)
lemmatized_new_sample = handle_negation([wnl.lemmatize(i[0], map_penn_to_wordnet(i[1])) for i in pos_tagged_new_sample])
joined_new_sample = ' '.join(lemmatized_new_sample)

sample_vectorized = loaded_vectorizer.transform([joined_new_sample])
loaded_model.predict(sample_vectorized)

def predict_sentiment(sentiment: str) -> int:
    tokenized_sentiment = word_tokenize(sentiment)
    lowercase_sentiment = [i.lower() for i in tokenized_sentiment]
    filtered_sentiment = [i for i in lowercase_sentiment if i not in stop_words]
    filtered_sentiment = [i for i in filtered_sentiment if i.isalpha() or i == 'n\'t']
    pos_tagged_sentiment = pos_tag(filtered_sentiment)
    lemmatized_sentiment = handle_negation([wnl.lemmatize(i[0], map_penn_to_wordnet(i[1])) for i in pos_tagged_sentiment])
    joined_sentiment = ' '.join(lemmatized_sentiment)

    sentiment_vectorized = loaded_vectorizer.transform([joined_sentiment])
    return loaded_model.predict(sentiment_vectorized)