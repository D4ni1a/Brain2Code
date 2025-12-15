def basic_checks_for_metrics(true_words, found_words):
    if len(true_words) == 0:
        raise ValueError("Sentence must contain at least one word!")
    if len(true_words) != len(found_words):
        raise ValueError(f"Number of words in a sentence ({len(true_words)}) is not equal to the number of found words ({len(found_words)})!")


def sentence_recall_k(true_words, found_words, k=None):
    basic_checks_for_metrics(true_words, found_words)
    n_found = 0
    cur_k = k
    for w_ind in range(len(true_words)):
        if k is None:
            cur_k = len(found_words[w_ind])
        if true_words[w_ind] in found_words[w_ind][:cur_k]:
            n_found += 1
    return n_found / len(true_words)

def sentence_MRR(true_words, found_words, k=None):
    """
    Mean reciprocal rank for words in a sentence
    """
    basic_checks_for_metrics(true_words, found_words)
    mrr = 0
    cur_k = k
    for w_ind in range(len(true_words)):
        if k is None:
            cur_k = len(found_words[w_ind])
        if true_words[w_ind] in found_words[w_ind][:cur_k]:
            mrr += 1 / ( found_words[w_ind][:cur_k].index(true_words[w_ind]) +1)
    return mrr / len(true_words)

    # 300
    # 30 *1500 = 45 000


