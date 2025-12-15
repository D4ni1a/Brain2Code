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

def run_experiment(model, classifier, train_loader, test_loader, num_epochs=25,
                   learning_rate=0.001, checkpoint_dir='checkpoints', device=None):
    """
    Run the complete training and evaluation experiment.

    Args:
        model: The main model
        classifier: The classifier head
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save model checkpoints
        device: Device to run computations on (CPU/GPU). If None, auto-detects.

    Returns:
        Dictionary with training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move models to device
    model = model.to(device)
    classifier = classifier.to(device)

    # Initialize loss function and optimizer
    # loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize training history
    history = {
        'train_loss': [],
        'train_cosine_sim': [],
        'test_loss': [],
        'test_cosine_sim': [],
        'best_epoch': 0
    }

    # Track best model
    best_test_cosine_sim = -float('inf')
    test_loss, test_cosine_sim = test_epoch(
        model, classifier, test_loader, loss_fn, device
    )
    history['test_loss'].append(test_loss)
    history['test_cosine_sim'].append(test_cosine_sim)


    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        train_loss, train_cosine_sim = train_epoch(
            model, classifier, train_loader, loss_fn, optimizer, device
        )

        # Testing phase
        test_loss, test_cosine_sim = test_epoch(
            model, classifier, test_loader, loss_fn, device
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_cosine_sim'].append(train_cosine_sim)
        history['test_loss'].append(test_loss)
        history['test_cosine_sim'].append(test_cosine_sim)

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training - Loss: {train_loss:.4f}, CosSim: {-train_cosine_sim:.4f}")
        print(f"  Testing  - Loss: {test_loss:.4f}, CosSim: {test_cosine_sim:.4f}")

        # Save best model
        if test_cosine_sim > best_test_cosine_sim:
            best_test_cosine_sim = test_cosine_sim
            history['best_epoch'] = epoch

            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_cosine_sim': test_cosine_sim,
                'history': history
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint_path)
    print(f"\nSaved final model to {final_checkpoint_path}")

    return history



# if __name__ == "__main__":
#     history = run_experiment(
#         model=model,
#         classifier=classifier,
#         train_loader=all_sets['train_loader'],
#         test_loader=all_sets['test_loader'],
#         num_epochs=200,
#         learning_rate=0.0001,
#         checkpoint_dir='checkpoints',
#         device=None  # Auto-detect device
#     )
