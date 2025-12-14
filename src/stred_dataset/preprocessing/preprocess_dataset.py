import numpy as np
import torch
from datasets import load_dataset
from typing import List, Dict, Optional
import warnings
import pickle
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.stred_dataset.preprocessing.channels_reordering import reorder_eeg_channels, make_channels_reordering, \
    get_basic_channels
from src.stred_dataset.preprocessing.eeg_downsapling import downsample_with_zoom
from src.stred_dataset.word_embedding import make_embedding_cache, get_word_embedding

warnings.filterwarnings('ignore')

def process_single_sample(sample: Dict,
                          index: int,
                          selected_channels_indices: List[int],
                          embedding_cache: Dict) -> Dict:
    """
    Processes a single datapoint
    """
    # Extract data
    word = sample['word']
    eeg_data = np.array(sample['eeg'])

    # 1. Downsampling
    eeg_downsampled = downsample_with_zoom(eeg_data, 200)

    # 2. Channel reordering
    eeg_processed = reorder_eeg_channels(eeg_downsampled, selected_channels_indices)

    # 3. Unsqueeze number of patches dimension
    eeg_processed = np.expand_dims(eeg_processed, axis=1)

    # 4. Get embedding
    word_embedding = get_word_embedding(word, embedding_cache)

    # 5. Save index and additional data
    result = {
        'index': index,
        'word': word,
        'word_emb': word_embedding,
        'eeg_data': eeg_processed.astype(np.float32),
        'topic': sample.get('topic', ''),
        'participant': sample.get('participant', ''),
    }

    return result


def preprocess_and_save(dataset_name: str = "Quoron/EEG-semantic-text-relevance",
                       all_channels: List[str] = None,
                       selected_channels: List[str] = None,
                       save_dir: str = SAVE_DIR,
                       chunk_size: int = CHUNK_SIZE,
                       max_samples: Optional[int] = None,
):
    """
    Main preprocessing and data saving function
    """
    print("-" * 60)
    print("START DATA PREPROCESSING")

    # Create directory for saving
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset in streaming mode
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, "data", split="train", cache_dir="./original_data",
                        #    streaming=True
                           )

    # Preparation
    current_chunk = []
    chunk_index = 0
    total_processed = 0
    embedding_cache = {}
    selected_channels_indices = make_channels_reordering(all_channels, selected_channels)


    # Data processing
    for idx, sample in enumerate(dataset):
        # Limit number of samples (for testing)
        if max_samples and idx >= max_samples:
            break

        # Process single sample
        processed_sample = process_single_sample(
            sample=sample,
            index=idx,
            selected_channels_indices=selected_channels_indices,
            embedding_cache=embedding_cache
        )

        current_chunk.append(processed_sample)

        # Save chunk when reaching size limit
        if len(current_chunk) >= chunk_size:
            save_chunk(current_chunk, chunk_index, save_dir)

            # Memory cleanup
            current_chunk = []
            gc.collect()

            chunk_index += 1
            print(f"\nSaved chunk {chunk_index}")

        total_processed += 1

        # Periodic info output
        if total_processed % 1000 == 0:
            print(f"\nProcessed: {total_processed} samples")
            print(f"Embedding cache size: {len(embedding_cache)}")

    # Save last chunk
    if current_chunk:
        save_chunk(current_chunk, chunk_index, save_dir)
        print(f"\nSaved last chunk {chunk_index}")

    # Save metadata
    save_metadata(total_processed, chunk_index + 1, chunk_size, save_dir, all_channels, selected_channels)

    print("\n" + "-" * 60)
    print(f"PREPROCESSING COMPLETED")
    print(f"Total processed: {total_processed} samples")
    print(f"Chunks saved: {chunk_index + 1}")
    print(f"Directory: {save_dir}")
    print("-" * 60)

def save_chunk(chunk_data: List[Dict], chunk_id: int, save_dir: str):
    """
    Saves chunk data to file
    """
    filename = os.path.join(save_dir, f"chunk_{chunk_id:04d}.pt")

    # Convert to PyTorch tensors
    processed_chunk = []
    for sample in chunk_data:
        processed_sample = {
            'index': torch.tensor(sample['index'], dtype=torch.long),
            'word': sample['word'],
            'word_emb': torch.tensor(sample['word_emb'], dtype=torch.float32),
            'eeg_data': torch.tensor(sample['eeg_data'], dtype=torch.float32),
            'topic': sample['topic']
        }
        processed_chunk.append(processed_sample)

    # Save in PyTorch format
    torch.save(processed_chunk, filename)

    # Also save in numpy format for compatibility
    numpy_filename = os.path.join(save_dir, f"chunk_{chunk_id:04d}_meta.pkl")
    chunk_meta = {
        'indices': [s['index'] for s in chunk_data],
        'words': [s['word'] for s in chunk_data],
        'size': len(chunk_data)
    }
    with open(numpy_filename, 'wb') as f:
        pickle.dump(chunk_meta, f)

def save_metadata(total_samples: int, total_chunks: int, chunk_size: int, save_dir: str,
                  all_channels: List[str] = None,
                  selected_channels: List[str] = None,
                  ):
    """
    Saves metadata about preprocessed data
    """
    metadata = {
        'total_samples': total_samples,
        'total_chunks': total_chunks,
        'chunk_size': chunk_size,
        'chunk_files': [f"chunk_{i:04d}.pt" for i in range(total_chunks)],
        'data_shape': {
            'eeg_data': (19, 200),  # After processing
            'word_emb': (300,)
        },
        'channels_info': {
            'all_channels': all_channels,
            'selected_channels': selected_channels,
            'n_selected': len(selected_channels)
        },
        'processing_info': {
            'downsampling': '2001 -> 200 time points',
            'word_embedding_dim': 300
        }
    }

    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nMetadata saved to {metadata_path}")

def process_sample_wrapper(args):
    """Wrapper for parallel processing"""
    sample, idx, selected_channels_indices, embedding_cache = args
    return process_single_sample(sample, idx, selected_channels_indices, embedding_cache)

def preprocess_parallel(dataset_name: str = "Quoron/EEG-semantic-text-relevance",
                       all_channels: List[str] = None,
                       selected_channels: List[str] = None,
                       save_dir: str = SAVE_DIR,
                       chunk_size: int = CHUNK_SIZE,
                       max_samples: int = None,
                       n_workers: int = 2,
                       embedding_cache = None):
    """
    Parallel version of preprocessing (for speedup)
    """
    print("Parallel processing...")

    os.makedirs(save_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, "data", split="train", cache_dir="./original_data")

    if embedding_cache is None:
        embedding_cache = {}
    current_chunk = []
    chunk_index = 0
    total_processed = 0
    selected_channels_indices = make_channels_reordering(all_channels, selected_channels)


    # Collect batch for parallel processing
    batch_for_processing = []

    for idx, sample in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        batch_for_processing.append((sample, idx, selected_channels_indices, embedding_cache))

        # Process batch
        if len(batch_for_processing) >= 100:  # Batch size for parallel processing
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_sample_wrapper, args)
                          for args in batch_for_processing]

                for future in as_completed(futures):
                    processed_sample = future.result()
                    current_chunk.append(processed_sample)

                    # Save chunk
                    if len(current_chunk) >= chunk_size:
                        save_chunk(current_chunk, chunk_index, save_dir)
                        current_chunk = []
                        chunk_index += 1

                    total_processed += 1

            batch_for_processing = []
            gc.collect()

            if total_processed % 1000 == 0:
                print(f"Processed: {total_processed}")

    # Process remainder
    if batch_for_processing:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_sample_wrapper, args)
                      for args in batch_for_processing]

            for future in as_completed(futures):
                processed_sample = future.result()
                current_chunk.append(processed_sample)
                total_processed += 1

    # Save last chunk
    if current_chunk:
        save_chunk(current_chunk, chunk_index, save_dir)

    save_metadata(total_processed, chunk_index + 1, chunk_size, save_dir, all_channels, selected_channels)
    print(f"Parallel processing completed: {total_processed} samples")


if __name__ == "__main__":
    save_dir = "processed_eeg_data"
    cache_dir = './original_data'
    all_channels, selected_channels, _ = get_basic_channels()

    data = load_dataset("Quoron/EEG-semantic-text-relevance", cache_dir=cache_dir)
    words = data['train']['word']

    embedding_cache = make_embedding_cache(words)
    preprocess_parallel(
        dataset_name = "Quoron/EEG-semantic-text-relevance",
        all_channels = all_channels,
        selected_channels = selected_channels,
        save_dir = save_dir,
        chunk_size = 5000,
        max_samples=None,
        n_workers=5,
        embedding_cache=embedding_cache
    )