import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

from src.stred_dataset.pytorch_dataset import EEGTextDataset
from src.stred_dataset.preprocessing.samples_filtering import get_filtered_ids


def make_dataset_split_make_dataloaders(data_dir="processed_eeg_data",
                                        max_samples=None,
                                        test_size=0.2,
                                        val_size=0.1,
                                        batch_size=32,
                                        shuffle_train=True,
                                        seed=42,
                                        filtered_indices=None):
    """
    Разделяет датасет на train/validation/test и создает DataLoader'ы

    Args:
        dataset: Экземпляр EEGTextDataset
        test_size: Доля тестовой выборки (от 0 до 1)
        val_size: Доля валидационной выборки от train (от 0 до 1)
        batch_size: Размер батча
        shuffle_train: Перемешивать ли train выборку
        seed: Seed для воспроизводимости

    Returns:
        Словарь с DataLoader'ами и индексами разделения
    """
    dataset = EEGTextDataset(
        data_dir=data_dir,
        max_samples=None,
    )


    # Получаем общее количество сэмплов
    total_samples = len(dataset)
    if filtered_indices is None:
        indices = list(range(total_samples))
    else:
        indices = filtered_indices

    if max_samples:
        indices = indices[:max_samples]

    # Разделяем на train+val и test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # Разделяем train+val на train и val
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=seed,
            shuffle=True
        )
    else:
        train_idx = train_val_idx
        val_idx = []

    print(f"Разделение датасета:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/total_samples:.1%})")
    print(f"  Validation: {len(val_idx)} samples ({len(val_idx)/total_samples:.1%})")
    print(f"  Test: {len(test_idx)} samples ({len(test_idx)/total_samples:.1%})")

    # Создаем Subset'ы
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Функция для коллации батча
    def collate_fn(batch):
        """Custom function for batch collation"""
        collated = {
            'index': torch.stack([item['index'] for item in batch]),
            'word': [item['word'] for item in batch],
            'word_emb': torch.stack([item['word_emb'] for item in batch]),
            'eeg_data': torch.stack([item['eeg_data'] for item in batch]),
            'topic': [item.get('topic', '') for item in batch],
            # 'participant': [item.get('participant', '') for item in batch]
        }
        return collated

    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=5,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Обычно validation не перемешивают
        num_workers=5,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test обычно не перемешивают
        num_workers=5,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'indices': {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    }

if __name__ == "__main__":
    selected_indices = get_filtered_ids(cache_dir='./original_data',
                     skipped_topics=['bank', 'bill clinton', 'football', 'euro', 'plato', 'schizophrenia', 'wine'],
                     limit_eeg_value=100,
                     limit_word_count=50)
    all_sets = make_dataset_split_make_dataloaders(data_dir="./processed_eeg_data",
                                        max_samples=None,
                                        test_size=0.2,
                                        val_size=0.,
                                        batch_size=128,
                                        shuffle_train=True,
                                        seed=42,
                                        filtered_indices=selected_indices)