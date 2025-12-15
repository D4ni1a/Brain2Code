import numpy as np
from annoy import AnnoyIndex
import torch

def get_vector_index(embeddings):
    """
    Creating Annoy index based on embedded data

    :return: Annoy index
    """
    # Input dimension = 768
    idx = AnnoyIndex(768, 'angular')# 'dot'
    print("Building an index...")
    for i in tqdm(range(embeddings.shape[0])):
        idx.add_item(i, embeddings[i])
    print("Building trees...")
    idx.build(51)
    idx.save('./my_annoy.ann')
    print("Finished!")
    return idx

def get_kNN_embeddings(embedding, k, index):
    """
    Creating Annoy index based on embedded data

    :param embedding: vector of embeddid text
    :param k: number of neighbors
    :param index: Annoy index
    :return: ids of neighbors
    """
    return index.get_nns_by_vector(embedding, k)

def make_eeg_embeddings(model, classifier, dataset, selected_indices, topics_distr):
    model.eval()
    classifier.eval()
    model = model.to('cpu')
    classifier = classifier.to('cpu')

    eeg_dataset = []
    eeg_embeds = []
    for ind in selected_indices:
        eeg_dataset.append(dataset[ind]['eeg_data'])

        emb = classifier(model(torch.from_numpy(np.array([eeg_dataset[-1]])) / 100.0 ))
        eeg_embeds.append(emb.detach().numpy() )

    eeg_dataset = np.array(eeg_dataset)
    eeg_embeds = np.array(eeg_embeds)

    word_set_in_index = topics_distr.loc[selected_indices]['word'].values
    return eeg_dataset, eeg_embeds, word_set_in_index

def find_closed_words(sent_embeddings, annoy_index, word_set_in_index, k = 5):
    # Search emb
    found_words = []
    for i in range(len(sent_embeddings)):
        found_words.append([])
        query = sent_embeddings[i] + torch.randn(*sent_embeddings[i].shape) / 20
        if len(query) != 0:
            idx = get_kNN_embeddings(query, k, annoy_index)
            # idx = get_kNN_embeddings(embed(query, spacy_model, method="spacy"), k, embedding_index)
            print(f"{i}:")
            for j in range(k):
                print(f"\t- {word_set_in_index[idx[j]]}")
                found_words[-1].append(word_set_in_index[idx[j]])
            print("=" * 30)
    return found_words
