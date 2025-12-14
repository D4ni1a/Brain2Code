import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from typing import List, Dict, Optional
import numpy as np

class EEGTextDataset(Dataset):
    """
    Dataset for working with preprocessed EEG data
    All data is loaded into memory for maximum speed
    """

    def __init__(self,
                 data_dir: str = "processed_eeg_data",
                 max_samples: Optional[int] = None,
                 preload_all: bool = True):
        """
        Args:
            data_dir: Directory with preprocessed data
            max_samples: Maximum number of samples (for testing)
            preload_all: If True, load all data into memory
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.preload_all = preload_all

        # Load metadata
        self.metadata = self._load_metadata()

        # Preload all data into memory if requested
        if self.preload_all:
            self._preload_all_data()
        else:
            # Initialize for lazy loading
            self.chunk_cache = {}
            self.chunk_access_order = []
            self.cache_hits = 0
            self.cache_misses = 0

        print(f"Initialized EEGTextDataset with {self.total_samples} samples")
        if self.preload_all:
            print("All data preloaded into memory")

    def _load_metadata(self) -> Dict:
        """Loads metadata from file"""
        metadata_path = os.path.join(self.data_dir, "metadata.pkl")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found in {metadata_path}. "
                f"First run the preprocessing script."
            )

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Save important fields as attributes
        self.total_samples = metadata['total_samples']
        self.total_chunks = metadata['total_chunks']
        self.chunk_size = metadata['chunk_size']
        self.chunk_files = metadata['chunk_files']
        self.data_shape = metadata['data_shape']

        # Limit number of samples
        if self.max_samples:
            self.total_samples = min(self.total_samples, self.max_samples)

        return metadata

    def _preload_all_data(self):
        """Preloads all data into memory"""
        print(f"Preloading all data from {self.total_chunks} chunks...")

        # Calculate how many chunks we need to load
        chunks_to_load = min(
            self.total_chunks,
            (self.total_samples + self.chunk_size - 1) // self.chunk_size
        )

        # Load all chunks
        self.all_samples = []
        for chunk_id in range(chunks_to_load):
            chunk_data = self._load_chunk(chunk_id)

            # Convert numpy arrays to torch tensors immediately
            for sample in chunk_data:
                if isinstance(sample['eeg_data'], np.ndarray):
                    sample['eeg_data'] = torch.from_numpy(sample['eeg_data']).float()
                if isinstance(sample['word_emb'], np.ndarray):
                    sample['word_emb'] = torch.from_numpy(sample['word_emb']).float()
                if 'index' in sample and isinstance(sample['index'], np.ndarray):
                    sample['index'] = torch.from_numpy(sample['index']).long()
                elif 'index' in sample and isinstance(sample['index'], (int, np.integer)):
                    sample['index'] = torch.tensor(sample['index'], dtype=torch.long)

            self.all_samples.extend(chunk_data)

        # Trim to max_samples if specified
        if self.max_samples and len(self.all_samples) > self.max_samples:
            self.all_samples = self.all_samples[:self.max_samples]
            self.total_samples = len(self.all_samples)

        print(f"Preloaded {len(self.all_samples)} samples into memory")

    def _load_chunk(self, chunk_id: int) -> List[Dict]:
        """Loads chunk from file"""
        if chunk_id >= len(self.chunk_files):
            raise IndexError(f"Chunk {chunk_id} does not exist")

        chunk_path = os.path.join(self.data_dir, self.chunk_files[chunk_id])

        if not os.path.exists(chunk_path):
            # Try alternative format
            alt_path = chunk_path.replace('.pt', '.pkl')
            if os.path.exists(alt_path):
                chunk_path = alt_path
            else:
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        # Load data
        if chunk_path.endswith('.pt'):
            chunk_data = torch.load(chunk_path, map_location='cpu')
        else:
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)

        return chunk_data

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Dict:
        """Gets sample by index"""
        if self.preload_all:
            # Direct access from memory
            return self.all_samples[index]
        else:
            # Original lazy loading logic (for comparison)
            return self._get_item_lazy(index)

    def _get_item_lazy(self, index: int) -> Dict:
        """Gets sample by index using lazy loading"""
        chunk_id = index // self.chunk_size
        pos_in_chunk = index % self.chunk_size

        # Get chunk (from cache or disk)
        if chunk_id in self.chunk_cache:
            chunk_data = self.chunk_cache[chunk_id]
            self.chunk_access_order.remove(chunk_id)
            self.chunk_access_order.append(chunk_id)
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            chunk_data = self._load_chunk(chunk_id)

            if len(self.chunk_cache) >= 2:  # Keep 2 chunks in cache
                oldest_chunk = self.chunk_access_order.pop(0)
                del self.chunk_cache[oldest_chunk]

            self.chunk_cache[chunk_id] = chunk_data
            self.chunk_access_order.append(chunk_id)

        # Get specific sample
        sample = chunk_data[pos_in_chunk]

        # Convert to required format
        if isinstance(sample['eeg_data'], np.ndarray):
            sample['eeg_data'] = torch.from_numpy(sample['eeg_data']).float()
        if isinstance(sample['word_emb'], np.ndarray):
            sample['word_emb'] = torch.from_numpy(sample['word_emb']).float()
        if 'index' in sample and isinstance(sample['index'], np.ndarray):
            sample['index'] = torch.from_numpy(sample['index']).long()
        elif 'index' in sample and isinstance(sample['index'], (int, np.integer)):
            sample['index'] = torch.tensor(sample['index'], dtype=torch.long)

        return sample

    def get_cache_stats(self) -> Dict:
        """Returns cache usage statistics (only for lazy loading)"""
        if self.preload_all:
            return {
                'preloaded': True,
                'total_samples': len(self.all_samples)
            }

        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_chunks': list(self.chunk_cache.keys()),
            'cache_size': len(self.chunk_cache),
            'preloaded': False
        }

    def clear_cache(self):
        """Clears chunk cache (only for lazy loading)"""
        if not self.preload_all:
            self.chunk_cache.clear()
            self.chunk_access_order.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print("Chunk cache cleared")

    def get_sample_info(self, index: int) -> Dict:
        """Returns sample information without loading data"""
        chunk_id = index // self.chunk_size
        pos_in_chunk = index % self.chunk_size

        # Load chunk metadata if available
        meta_path = os.path.join(self.data_dir, f"chunk_{chunk_id:04d}_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                chunk_meta = pickle.load(f)

            if pos_in_chunk < len(chunk_meta['indices']):
                return {
                    'chunk_id': chunk_id,
                    'position': pos_in_chunk,
                    'word': chunk_meta['words'][pos_in_chunk],
                    'original_index': chunk_meta['indices'][pos_in_chunk]
                }

        return {
            'chunk_id': chunk_id,
            'position': pos_in_chunk,
            'original_index': index
        }

def check_processed_data(data_dir: str = "processed_eeg_data"):
    """Checks integrity of preprocessed data"""
    print(f"Checking data in {data_dir}...")

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        return False

    # Check metadata
    metadata_path = os.path.join(data_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found")
        return False

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Metadata loaded:")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Chunks: {metadata['total_chunks']}")
    print(f"  Chunk size: {metadata['chunk_size']}")
    print(f"  EEG shape: {metadata['data_shape']['eeg_data']}")

    # Check if chunk files exist
    missing_files = []
    for i in range(min(5, metadata['total_chunks'])):  # Check first 5
        chunk_file = f"chunk_{i:04d}.pt"
        if not os.path.exists(os.path.join(data_dir, chunk_file)):
            missing_files.append(chunk_file)

    if missing_files:
        print(f"Missing files: {missing_files}")
        return False

    print(f"All files present")
    return True


if __name__ == "__main__":
    check_processed_data()

    dataset = EEGTextDataset(
        data_dir= "./processed_eeg_data",
        max_samples=None,
    )