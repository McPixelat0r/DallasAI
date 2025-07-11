import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk

# pd.set_option('display.max_columns', None)

# --- NLTK Data Downloads (Run this ONCE, then comment out or remove) ---
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_en')

# ---------------------------------------------------------------------

_NEGATING_WORDS = {'not', 'no', 'n\'t', 'never', 'none', 'neither', 'nor', 'hardly', 'barely', 'scarcely', 'seldom'}
_STOP_WORDS = set(stopwords.words('english'))
_wnl = nltk.stem.WordNetLemmatizer()


def _map_penn_to_wordnet(tag: str) -> str:
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


def _handle_negation(token_lst: list) -> list:
    i = 0
    processed_tokens = []
    while i < len(token_lst):
        token = token_lst[i]
        if token_lst[i] in _NEGATING_WORDS:
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


def _filter_tokens(token_lst: list[str]) -> list[str]:
    filtered_tokens = list[str]
    token_index: int = 0
    while token_index < len(token_lst):
        current_token: str = token_lst[token_index].lower()
        if current_token not in stopwords and (current_token.isalpha() or current_token == 'n\'t'):
            filtered_tokens.append(current_token)

    return filtered_tokens


def _lemmatize_tokens(pos_tagged_tokens: list[(str, str)]) -> list[str]:
    lematized_tokens = list[str]
    for pos_tagged_token in pos_tagged_tokens:
        lematized_tokens.append(_wnl.lemmatize(pos_tagged_token[0], _map_penn_to_wordnet(pos_tagged_token[1])))
    return lematized_tokens


def preprocess_data(raw_data: pd.DataFrame):
    tokenized_data: pd.Series = raw_data['feedback_text'].apply(word_tokenize)
    filtered_tokens: pd.Series = tokenized_data.apply(_filter_tokens)
    pos_tagged_tokens: pd.Series = filtered_tokens.apply(pos_tag)
    lemmatized_tokens: pd.Series = pos_tagged_tokens.apply(_lemmatize_tokens)
    tokenized_data = lemmatized_tokens.apply(_handle_negation)

    return tokenized_data


def main():
    sample_feedback_data: dict = {'feedback_id': [1, 2, 3, 4],
                                  'ai_tool_name': ['ChatGPT 4.0', 'Midjourney', 'Copilot', 'Internal Tool X'],
                                  'user_id': ['user_A', 'user_B', 'user_C', 'user_D'],
                                  'timestamp': ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04'],
                                  'feedback_text': [
                                      'The interface is super intuitive, but it crashes sometimes during heavy use',
                                      'Image generation is amazing, but the commands are confusing. I wish it was '
                                      'more '
                                      'stable.',
                                      'Great for code suggestions, really speeds up my work. Never had an issue '
                                      'with '
                                      'stability.',
                                      'Hard to navigate, very slow, and keeps freezing. Not worth the effort.'],
                                  'sentiment': ['neutral', 'neutral', 'positive', 'negative']}

    df = pd.DataFrame.from_records(sample_feedback_data)

    print(preprocess_data(df))


if __name__ == '__main__':
    main()
