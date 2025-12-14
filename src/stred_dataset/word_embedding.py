import nltk
nltk.download('wordnet')
nltk.download('omw-eng')
# nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import scipy.signal as signal
from scipy.ndimage import zoom
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.ndimage import zoom
from sentence_transformers import SentenceTransformer

BERT_MODEL = SentenceTransformer('bert-base-nli-mean-tokens')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_word(word, stem=True, lemma=True):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # if stem:
    #     word = stemmer.stem(word)
    if lemma:
        word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    return word

def embed(word, model, method=""):
    """
    Normalized text embedding

    :param word: string text
    :param model: embedding model
    :return: vector of embedding
    """
    text = preprocess_word(word)
    if method == "spacy":
        vect = np.array(model.vocab[text].vector)
    else: # bert
        vect = model.encode(text, show_progress_bar=False)
    return vect / np.linalg.norm(vect)

def get_word_embedding(word: str, cache: Optional[Dict] = None,  model=BERT_MODEL, method="bert") -> np.ndarray:
    """
    Retrieves a word embedding vector of length 300.
    IMPORTANT: Implement this function according to your pre-trained model.

    Args:
        word: Word to vectorize
        model: Pre-trained model (if available)
        cache: Embedding cache for speed-up

    Returns:
        Embedding vector of size 300
    """
    if cache is not None and word in cache:
        return cache[word]

    # Make emb for word not in cache
    embedding = embed(word, model, method=method)

    if cache is not None:
        cache[word] = embedding
    return embedding

def make_embedding_cache(words):
    # Make embedding hash
    embedding_cache = {}
    for word in words:
        embedding_cache[word] = embed(word, BERT_MODEL, method="bert")
    return embedding_cache