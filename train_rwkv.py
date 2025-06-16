# ==============================================================================
# IMPORTANT: Before running this script, you must compile the custom CUDA kernel.
# In your terminal, run the following command from the project's root directory:
#
# python setup.py install
#
# This will build and install the 'custom_wkv_kernel' module, making it
# available for import. This script relies on that pre-compiled module.
# ==============================================================================

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

import wandb

# --- Import RWKV-7 Model ---
# This assumes 'rwkv_model.py' is in the same directory.
try:
    from rwkv_model import RWKV7_Model_Classifier
except ImportError:
    print("FATAL: Could not import RWKV7_Model_Classifier from rwkv_model.py.")
    print("Please ensure rwkv_model.py is in the same directory as this script.")
    exit()

# --- Custom CUDA Kernel Check ---
# Instead of compiling on the fly, we now check if the pre-compiled module is available.
try:
    import custom_wkv_kernel
    import rwkv_model
    rwkv_model.use_custom_kernel = True
    print("Successfully imported pre-compiled 'custom_wkv_kernel'. Using CUDA kernel.")
    # Log to wandb if a run is active
    if wandb.run:
        wandb.log({"cuda_kernel_status": "enabled"})
except ImportError:
    print("\n" + "="*70)
    print("WARNING: Could not import the 'custom_wkv_kernel' module.")
    print("This means the CUDA kernel has not been compiled and installed correctly.")
    print("Please make sure you have run 'python setup.py install' in your environment.")
    print("Falling back to the pure PyTorch implementation. This will be significantly slower.")
    print("="*70 + "\n")
    import rwkv_model
    rwkv_model.use_custom_kernel = False
    if wandb.run:
        wandb.log({"cuda_kernel_status": "disabled"})


# --- Data Handling ---

class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, alphabet):
        self.alphabet = sorted(list(set(alphabet)))
        self.char_to_token = {c: i+1 for i, c in enumerate(self.alphabet)}
        self.char_to_token['<pad>'] = 0
        self.token_to_char = {i: c for c, i in self.char_to_token.items()}
        self.vocab_size = len(self.char_to_token)

    def encode(self, s):
        return [self.char_to_token[c] for c in s]

    def decode(self, t):
        return "".join([self.token_to_char[i] for i in t])

class RegularLanguageDataset(Dataset):
    """PyTorch Dataset for loading regular language sequences."""
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['string']  # Changed 'sequence' to 'string'
        # Handle potential float (NaN) values for empty sequences
        if not isinstance(sequence, str):
            sequence = ''
        
        tokens = self.tokenizer.encode(sequence)
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float)
        }

def collate_fn(batch):
    """Pads sequences in a batch to the maximum length in that batch."""
    tokens = [item['tokens'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    max_len = max(len(t) for t in tokens) if tokens else 0
    
    padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t
        
    return {'tokens': padded_tokens, 'labels': labels}


# --- Training and Evaluation Logic (Refactored into helper functions) ---

def evaluate(model, data_loader, device, loss_fn):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(tokens) # states are not needed for eval
            loss = loss_fn(logits.squeeze(-1), labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).squeeze(-1) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def _setup_dataloaders(config, tokenizer, batch_size=None):
    """Helper to create and return data loaders."""
    if batch_size is None:
        batch_size = config.batch_size
        
    base_path = os.path.join("datasets", config.lang, f"train_{config.train_len}_test_{config.val_test_lens[0]}_{config.val_test_lens[1]}")
    train_dataset = RegularLanguageDataset(os.path.join(base_path, "train.csv"), tokenizer)
    val_dataset = RegularLanguageDataset(os.path.join(base_path, "validation.csv"), tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # Return validation loaders as a dictionary to match expected format
    val_loaders = {f"val_{config.val_test_lens[0]}": val_loader}
    
    return train_loader, val_loaders, base_path

def _train_epoch(model, train_loader, optimizer, loss_fn, device, scaler):
    """Helper for a single training epoch."""
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits, _ = model(tokens)
            loss = loss_fn(logits.squeeze(-1), labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Unscale for clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

def _validate_and_checkpoint(model, val_loaders, device, loss_fn, config, epoch, avg_train_loss, best_val_loss, patience_counter, step_offset=0):
    """Helper for validation, logging, and checkpointing with early stopping."""
    val_metrics = {}
    for name, loader in val_loaders.items():
        metrics = evaluate(model, loader, device, loss_fn)
        val_metrics.update({f"{name}_{k}": v for k, v in metrics.items()})

    # Use step_offset to ensure epochs start from 0 on retry
    log_data = {'epoch': epoch, 'train_loss': avg_train_loss, **val_metrics}
    wandb.log(log_data, step=epoch + step_offset)

    val_loss_key = f'val_{config.val_test_lens[0]}_loss' if config.val_test_lens else None
    val_loss_for_log = val_metrics.get(val_loss_key, 'N/A') if val_loss_key else 'N/A'
    print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_for_log if isinstance(val_loss_for_log, str) else f'{val_loss_for_log:.4f}'}")

    current_val_loss = val_metrics.get(val_loss_key, float('inf')) if val_loss_key else float('inf')

    # Early stopping logic
    improved = False
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0  # Reset patience counter
        improved = True
        if wandb.run:
            model_path = os.path.join(wandb.run.dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            wandb.save("best_model.pt") # Use relative path for files in wandb.run.dir
        print(f"New best validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")
    
    return best_val_loss, patience_counter, improved

def _final_test_evaluation(model, config, base_path, tokenizer, device, loss_fn, batch_size=None):
    """Helper for final evaluation on test sets."""
    if batch_size is None:
        batch_size = config.batch_size
        
    print("\n--- Running Final Test Evaluation ---")
    test_loaders = {}
    if config.val_test_lens:
        for l_test in config.val_test_lens:
            test_path = os.path.join(base_path, f"test_{l_test}.csv")
            if os.path.exists(test_path):
                test_dataset = RegularLanguageDataset(test_path, tokenizer)
                test_loaders[f"test_{l_test}"] = DataLoader(
                    test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
                )
    
    test_metrics = {}
    if wandb.run:
        best_model_path = os.path.join(wandb.run.dir, "best_model.pt")
        if os.path.exists(best_model_path):
            print("Loading best model from checkpoint for final evaluation.")
            model.load_state_dict(torch.load(best_model_path))
    else:
        print("WARNING: wandb.run is not active. Cannot load best model.")

    for name, loader in test_loaders.items():
        metrics = evaluate(model, loader, device, loss_fn)
        test_metrics.update({f"{name}_{k}": v for k, v in metrics.items()})
        print(f"Results for {name}: {metrics}")
    
    if test_metrics:
        wandb.log({"final_test_metrics": test_metrics})


def train_experiment(config):
    """Main function to run a single training experiment."""
    run_name = f"{config['lang']}_train{config['train_len']}_exp{config['exp_id']}"
    
    with wandb.init(
        project="regex-learning", 
        config=config,
        name=run_name,
        group=f"{config['lang']}-train{config['train_len']}",
        settings=wandb.Settings(init_timeout=900)
    ) as run:
        config = run.config # Use wandb.config for sweeps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Starting Experiment: {run_name} on {device} ---")
        print(f"Cuda kernel status: {'enabled' if rwkv_model.use_custom_kernel else 'disabled'}")
        print(f"Config: {config}")

        alphabets = {'L1': ['a', 'b'], 'L2': ['a', 'b'], 'L3': ['a', 'b', 'c'], 'L4': ['a', 'b', 'c']}
        tokenizer = CharTokenizer(alphabets[config.lang])
        # Start with the original batch size and handle OOM by reducing it
        current_batch_size = config.batch_size
        min_batch_size = 64
        batch_size_reduction = 64  # Reduce by 64 each time
        
        model_params = {
            'd_model': config.d_model, 'n_layer': config.n_layer,
            'vocab_size': tokenizer.vocab_size, 'head_size': config.head_size,
            'ffn_hidden_multiplier': config.ffn_hidden_multiplier,
            'lora_dim_w': config.lora_dim_w, 'lora_dim_a': config.lora_dim_a,
            'lora_dim_v': config.lora_dim_v, 'lora_dim_g': config.lora_dim_g,
        }
        
        # Retry loop for handling OOM errors
        training_successful = False
        step_offset = 0  # Initialize step offset for wandb logging
        while not training_successful and current_batch_size >= min_batch_size:
            try:
                print(f"Attempting training with batch size: {current_batch_size}")
                
                # Clear any existing GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Create data loaders with current batch size
                train_loader, val_loaders, base_path = _setup_dataloaders(config, tokenizer, current_batch_size)
                
                # Create model and move to device
                model = RWKV7_Model_Classifier(**model_params).to(device)
                print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
                run.watch(model, log='all', log_freq=100)

                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.get('weight_decay', 0.01))
                loss_fn = nn.BCEWithLogitsLoss()
                scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

                # Log the actual batch size being used
                wandb.log({"actual_batch_size": current_batch_size}, step=step_offset)
                if current_batch_size != config.batch_size:
                    print(f"WARNING: Reduced batch size from {config.batch_size} to {current_batch_size} due to memory constraints")
                    wandb.log({"batch_size_reduced": True, "original_batch_size": config.batch_size}, step=step_offset)

                # Early stopping parameters
                best_val_loss = float('inf')
                patience_counter = 0
                patience = config.get('patience', 5)
                min_epochs = config.get('min_epochs', 10)
                
                print(f"Early stopping enabled: patience={patience}, min_epochs={min_epochs}")
                  # Training loop
                for epoch in range(config.epochs):
                    avg_train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
                    best_val_loss, patience_counter, improved = _validate_and_checkpoint(
                        model, val_loaders, device, loss_fn, config, epoch, avg_train_loss, best_val_loss, patience_counter, step_offset
                    )
                    
                    # Check for early stopping
                    if epoch >= min_epochs and patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience={patience})")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                        wandb.log({"early_stopped_epoch": epoch + 1, "best_val_loss": best_val_loss}, step=epoch + step_offset)
                        break
                else:
                    print(f"\nTraining completed all {config.epochs} epochs")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                
                _final_test_evaluation(model, config, base_path, tokenizer, device, loss_fn, current_batch_size)
                  # If we reach here, training was successful
                training_successful = True
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"\nCUDA Out of Memory Error with batch size {current_batch_size}: {str(e)}")
                
                # Update step_offset to continue from where we left off
                if 'epoch' in locals():
                    step_offset += epoch + 1  # Add the epochs completed in this attempt
                
                # Clean up current model and data loaders
                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                if 'train_loader' in locals():
                    del train_loader, val_loaders
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Reduce batch size by 64
                current_batch_size = max(current_batch_size - batch_size_reduction, min_batch_size)
                
                if current_batch_size < min_batch_size:
                    print(f"ERROR: Cannot reduce batch size below {min_batch_size}. Experiment failed.")
                    wandb.log({"experiment_failed": True, "failure_reason": "batch_size_too_small"})
                    return
                else:
                    print(f"Retrying with reduced batch size: {current_batch_size} (reduced by {batch_size_reduction})")
                    wandb.log({
                        "oom_error": True, 
                        "reduced_batch_size_to": current_batch_size,
                        "batch_size_reduction": batch_size_reduction,
                        "oom_error_message": str(e)
                    })
        
        if not training_successful:
            print(f"ERROR: Training failed even with minimum batch size {min_batch_size}")
            wandb.log({"experiment_failed": True, "failure_reason": "persistent_oom"})
            return
        
        # Clean up
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if 'train_loader' in locals():
            del train_loader, val_loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Base config that applies to most experiments
    base_config = {
        'batch_size': 6144, 'epochs': 20,
        'ffn_hidden_multiplier': 4,
        'lora_dim_w': 32, 'lora_dim_a': 32,
        'lora_dim_v': 16, 'lora_dim_g': 32,
        'weight_decay': 0.01,
        'patience': 5,  # Early stopping patience
        'min_epochs': 10  # Minimum epochs before early stopping can trigger
    }

    LANG_TRAIN_CONFIGS = []
    for lang in ['L1', 'L2', 'L3', 'L4']:
        for train_len_config in [
            {'train_len': 50, 'val_test_lens': [100, 200]},
            {'train_len': 100, 'val_test_lens': [200, 300]}
        ]:
            LANG_TRAIN_CONFIGS.append({'lang': lang, **train_len_config})

    # print("\n\n===== STARTING EXPERIMENT 1: D_MODEL Sweep =====")
    # for lang_train_cfg in LANG_TRAIN_CONFIGS:
    #     for d_model in [10, 20, 30, 40, 50]:
    #         config = {
    #             **base_config, **lang_train_cfg, 'exp_id': f"2_d_model_{d_model}",
    #             'n_layer': 4, 'd_model': d_model, 'head_size': 10, 'learning_rate': 1e-4,
    #         }
    #         train_experiment(config)      


    # print("\n\n===== STARTING EXPERIMENT 2: D_MODEL and LR Sweep =====")
    # for lang_train_cfg in LANG_TRAIN_CONFIGS:
    #     for d_model in [80, 100]:
    #          for lr in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]:
    #             config = {
    #                 **base_config, **lang_train_cfg, 'exp_id': f"3_d_model_{d_model}_lr_{lr:.0e}",
    #                 'n_layer': 4, 'd_model': d_model, 'head_size': 10, 'learning_rate': lr,
    #             }
    #             train_experiment(config)

    print("\n\n===== STARTING EXPERIMENT 3: Pre-defined Model Sizes =====")
    exp1_configs = [
        {'exp_id': '1a_0.1B_arch', 'n_layer': 12, 'd_model': 768, 'head_size': 64, 'learning_rate': 6e-4},
        {'exp_id': '1b_0.4B_arch', 'n_layer': 24, 'd_model': 1024, 'head_size': 64, 'learning_rate': 5e-4},
    ]
    for lang_train_cfg in LANG_TRAIN_CONFIGS:
        for exp_cfg in exp1_configs:
            train_experiment({**base_config, **lang_train_cfg, **exp_cfg})

if __name__ == '__main__':
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log in to wandb: {e}\nPlease run 'wandb login' in your terminal.")
    main()
