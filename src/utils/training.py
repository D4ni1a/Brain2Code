import torch
import torch.optim as optim
import os

def train_epoch(model, classifier, train_loader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The main model
        classifier: The classifier head
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to run computations on (CPU/GPU)

    Returns:
        Average training loss, average training cosine similarity
    """
    model.train()
    classifier.train()

    epoch_loss = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'[Train]')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        data = batch["eeg_data"].to(device) / 100.0
        targets = batch["word_emb"].to(device)
        ys = torch.ones(data.shape[0], device=device)

        # Forward pass
        predictions = classifier(model(data))
        predictions = predictions / torch.linalg.norm(predictions, ord=2, dim=1, keepdim=True)
        targets = targets / torch.linalg.norm(targets, ord=2, dim=1, keepdim=True)
        # loss = loss_fn(predictions, targets, ys)
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_loss = loss.item()
        epoch_loss += batch_loss
        epoch_cosine_sim += loss.item()  # Note: CosineEmbeddingLoss returns -cosine_sim + margin
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'AvgLoss': f'{epoch_loss/num_batches:.4f}'
        })

    avg_train_loss = epoch_loss / num_batches
    avg_train_cosine_sim = epoch_cosine_sim / num_batches

    return avg_train_loss, avg_train_cosine_sim

def test_epoch(model, classifier, test_loader, loss_fn, device):
    """
    Evaluate the model on test data.

    Args:
        model: The main model
        classifier: The classifier head
        test_loader: DataLoader for test data
        loss_fn: Loss function
        device: Device to run computations on (CPU/GPU)

    Returns:
        Average test loss, average test cosine similarity
    """
    model.eval()
    classifier.eval()

    test_loss = 0.0
    test_cosine_sim = 0.0
    num_test_batches = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f'[Test]')
        for test_batch in test_pbar:
            # Move data to device
            test_data = test_batch["eeg_data"].to(device)
            test_targets = test_batch["word_emb"].to(device)
            test_ys = torch.ones(test_data.shape[0], device=device)

            # Forward pass
            test_predictions = classifier(model(test_data))
            test_predictions = test_predictions / torch.linalg.norm(test_predictions, ord=2, dim=1, keepdim=True)
            test_targets = test_targets / torch.linalg.norm(test_targets, ord=2, dim=1, keepdim=True)

            # test_batch_loss = loss_fn(test_predictions, test_targets, test_ys)
            test_batch_loss = loss_fn(test_predictions, test_targets)

            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(test_predictions, test_targets)
            avg_cos_sim = cos_sim.mean().item()

            # Update metrics
            test_loss += test_batch_loss.item()
            test_cosine_sim += avg_cos_sim
            num_test_batches += 1

            # Update progress bar
            test_pbar.set_postfix({
                'TestLoss': f'{test_batch_loss.item():.4f}',
                'TestCosSim': f'{avg_cos_sim:.4f}'
            })

    avg_test_loss = test_loss / num_test_batches
    avg_test_cosine_sim = test_cosine_sim / num_test_batches

    return avg_test_loss, avg_test_cosine_sim

