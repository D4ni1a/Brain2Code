from datasets import load_dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def get_filtered_ids(cache_dir='./original_data',
                     skipped_topics = ['bank','bill clinton', 'football', 'euro', 'plato', 'schizophrenia', 'wine'],
                     limit_eeg_value = 100,
                     limit_word_count = 50):


    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Load original data
    data = load_dataset("Quoron/EEG-semantic-text-relevance", cache_dir=cache_dir)

    # Get word - topic correspondence
    topics_distr = pd.DataFrame([data['train']['word'], data['train']['topic'], data['train']['participant']]).T
    topics_distr = topics_distr.rename(columns={0:"word", 1:"topic", 2:'participant'})
    print("Numer of topics:", len(topics_distr['topic'].unique()))

    # Drop not selected topics
    if skipped_topics is not None:
        filtered_indices = np.array(range(len(topics_distr)))[(~topics_distr['topic'].isin(skipped_topics) & topics_distr['word'].apply(str.isalpha)).values]
    else:
        filtered_indices = topics_distr.index.values

    # Filter indices with too big values
    if limit_eeg_value:
        s_ind = []
        for ind in filtered_indices:
            eeg_data = np.array(data['train']['eeg'][int(ind)])
            if np.abs(eeg_data).max() <= limit_eeg_value:
                s_ind.append(ind)
        filtered_indices = s_ind

    # Filter data to have at most 50 same words
    if limit_word_count:
        tmp_df = topics_distr.loc[filtered_indices]
        word_counts = tmp_df.groupby('word')['topic'].count()
        # Add words with < limit count occurrences
        selected_indices = tmp_df[tmp_df['word'].isin(word_counts[word_counts<=limit_word_count].index.to_numpy())].index.to_numpy()
        print("Number of unique words with occurrences count <= limit_word_count:", len(np.unique(word_counts[word_counts<=limit_word_count].index)))
        words_for_prunning = word_counts[word_counts>limit_word_count].index.to_numpy()
        print("Number of unique words with occurrences count > limit_word_count:", len(np.unique(words_for_prunning)))
        for word in words_for_prunning:
            word_df = tmp_df[tmp_df['word'] == word]

            try:
                _, word_df = train_test_split(
                    word_df,
                    test_size=limit_word_count/len(word_df),
                    stratify=word_df['topic'] ,
                    random_state=42
                )
                word_ids = word_df.index.values
            except:
                word_ids = np.random.choice(word_df.index.values, size=limit_word_count, replace=False)
            selected_indices = np.concatenate((selected_indices, word_ids))
        filtered_indices = selected_indices
    print("Total number of selected indices is:", len(filtered_indices))
    return filtered_indices

if __name__ == "__main__":
    filtered_indices = get_filtered_ids(cache_dir='./original_data',
                     skipped_topics=['bank', 'bill clinton', 'football', 'euro', 'plato', 'schizophrenia', 'wine'],
                     limit_eeg_value=100,
                     limit_word_count=50)