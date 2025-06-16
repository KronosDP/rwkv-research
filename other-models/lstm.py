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

# Add parent directory to path to import from train_rwkv.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_rwkv import (CharTokenizer, RegularLanguageDataset, collate_fn,
                        evaluate)

# Constants
BEST_MODEL_FILENAME = "best_model.pt"


class LSTMModel(nn.Module):
    """LSTM model for sequence classification."""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, tokens):
        # tokens: (batch_size, seq_len)
        embedded = self.embedding(tokens)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        last_output = self.dropout(last_output)
        
        # Classification
        logits = self.classifier(last_output)  # (batch_size, 1)
        
        return logits, (hidden, cell)


def _setup_dataloaders(config, tokenizer):
    """Helper to create and return data loaders."""
    base_path = os.path.join("datasets", config['lang'], f"train_{config['train_len']}_test_{config['val_test_lens'][0]}_{config['val_test_lens'][1]}")
    train_dataset = RegularLanguageDataset(os.path.join(base_path, "train.csv"), tokenizer)
    val_dataset = RegularLanguageDataset(os.path.join(base_path, "validation.csv"), tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # Return validation loaders as a dictionary to match expected format
    val_loaders = {f"val_{config['val_test_lens'][0]}": val_loader}
    
    return train_loader, val_loaders, base_path


def _train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Helper for a single training epoch."""
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)

        logits, _ = model(tokens)
        loss = loss_fn(logits.squeeze(-1), labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        
    return total_train_loss / len(train_loader) if len(train_loader) > 0 else 0


def _validate_and_checkpoint(model, val_loaders, device, loss_fn, config, epoch, avg_train_loss, best_val_loss, patience_counter, perfect_f1_counter):
    """Helper for validation, logging, and checkpointing with early stopping."""
    val_metrics = {}
    for name, loader in val_loaders.items():
        metrics = evaluate(model, loader, device, loss_fn)
        val_metrics.update({f"{name}_{k}": v for k, v in metrics.items()})

    log_data = {'epoch': epoch, 'train_loss': avg_train_loss, **val_metrics}
    wandb.log(log_data)

    val_loss_key = f'val_{config["val_test_lens"][0]}_loss' if config["val_test_lens"] else None
    val_f1_key = f'val_{config["val_test_lens"][0]}_f1' if config["val_test_lens"] else None
    val_loss_for_log = val_metrics.get(val_loss_key, 'N/A') if val_loss_key else 'N/A'
    val_f1_for_log = val_metrics.get(val_f1_key, 'N/A') if val_f1_key else 'N/A'
    
    print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_for_log if isinstance(val_loss_for_log, str) else f'{val_loss_for_log:.4f}'} | Val F1: {val_f1_for_log if isinstance(val_f1_for_log, str) else f'{val_f1_for_log:.4f}'}")

    current_val_loss = val_metrics.get(val_loss_key, float('inf')) if val_loss_key else float('inf')
    current_val_f1 = val_metrics.get(val_f1_key, 0.0) if val_f1_key else 0.0

    # Check for perfect F1 score
    if current_val_f1 >= 1.0:
        perfect_f1_counter += 1
        print(f"Perfect F1 score achieved! Count: {perfect_f1_counter}/2")
    else:
        perfect_f1_counter = 0  # Reset counter if F1 is not perfect

    # Early stopping logic
    improved = False
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0  # Reset patience counter
        improved = True
        if wandb.run:
            model_path = os.path.join(wandb.run.dir, BEST_MODEL_FILENAME)
            torch.save(model.state_dict(), model_path)
            wandb.save(BEST_MODEL_FILENAME)
        print(f"New best validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")
    
    return best_val_loss, patience_counter, improved, perfect_f1_counter


def _final_test_evaluation(model, config, base_path, tokenizer, device, loss_fn):
    """Helper for final evaluation on test sets."""
    print("\n--- Running Final Test Evaluation ---")
    test_loaders = {}
    if config["val_test_lens"]:
        for l_test in config["val_test_lens"]:
            test_path = os.path.join(base_path, f"test_{l_test}.csv")
            if os.path.exists(test_path):
                test_dataset = RegularLanguageDataset(test_path, tokenizer)
                test_loaders[f"test_{l_test}"] = DataLoader(
                    test_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, num_workers=0
                )
    
    test_metrics = {}
    if wandb.run:
        best_model_path = os.path.join(wandb.run.dir, BEST_MODEL_FILENAME)
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
    run_name = f"LSTM_{config['lang']}_train{config['train_len']}"
    
    # Add model type to config for wandb logging
    config['model_type'] = 'LSTM'
    
    with wandb.init(
        project="baseline", 
        config=config,
        name=run_name,
        group=f"LSTM-{config['lang']}-train{config['train_len']}"
    ) as run:
        config = run.config # Use wandb.config for sweeps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Starting LSTM Experiment: {run_name} on {device} ---")
        print(f"Config: {config}")

        alphabets = {'L1': ['a', 'b'], 'L2': ['a', 'b'], 'L3': ['a', 'b', 'c'], 'L4': ['a', 'b', 'c']}
        tokenizer = CharTokenizer(alphabets[config['lang']])
        train_loader, val_loaders, base_path = _setup_dataloaders(config, tokenizer)
        
        # Create LSTM model
        model = LSTMModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['embedding_dim'],  # Use same as embedding_dim
            num_layers=config['num_ff_layers'],
            dropout=config['dropout']
        ).to(device)
        print(f"LSTM model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
        run.watch(model, log='all', log_freq=100)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=0.01, momentum=0.9)
        loss_fn = nn.BCEWithLogitsLoss()        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        perfect_f1_counter = 0  # Counter for consecutive perfect F1 scores
        patience = config.get('patience', 10)
        min_epochs = config.get('min_epochs', 20)
        
        print(f"Early stopping enabled: patience={patience}, min_epochs={min_epochs}")
        print("F1 score early stopping: Will stop if F1=1.0 for 2 consecutive epochs")
        
        for epoch in range(config['epochs']):
            avg_train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
            best_val_loss, patience_counter, _, perfect_f1_counter = _validate_and_checkpoint(
                model, val_loaders, device, loss_fn, config, epoch, avg_train_loss, best_val_loss, patience_counter, perfect_f1_counter
            )
            
            # Check for perfect F1 early stopping
            if perfect_f1_counter >= 2:
                print(f"\nF1 score early stopping triggered after {epoch + 1} epochs (F1=1.0 for {perfect_f1_counter} consecutive epochs)")
                wandb.log({"f1_early_stopped_epoch": epoch + 1, "perfect_f1_count": perfect_f1_counter})
                break
            
            # Check for regular early stopping
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience={patience})")
                print(f"Best validation loss: {best_val_loss:.4f}")
                wandb.log({"early_stopped_epoch": epoch + 1, "best_val_loss": best_val_loss})
                break
        else:
            print(f"\nTraining completed all {config['epochs']} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        _final_test_evaluation(model, config, base_path, tokenizer, device, loss_fn)
        
        del model, optimizer, train_loader, val_loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main function to run LSTM experiments."""
    
    # LSTM configuration as specified
    base_config = {
        'learning_rate': 0.0001,
        'epochs': 1024,
        'embedding_dim': 256,
        'num_ff_layers': 2,
        'dropout': 0.0,
        'batch_size': 5120,
        'patience': 10,
        'min_epochs': 20
    }
    
    # Use same LANG_TRAIN_CONFIGS as RWKV
    LANG_TRAIN_CONFIGS = []
    for lang in ['L1', 'L2', 'L3', 'L4']:
        for train_len_config in [
            {'train_len': 50, 'val_test_lens': [100, 200]},
            {'train_len': 100, 'val_test_lens': [200, 300]}
        ]:
            LANG_TRAIN_CONFIGS.append({'lang': lang, **train_len_config})

    print("\n\n===== STARTING LSTM BASELINE EXPERIMENTS =====")
    for lang_train_cfg in LANG_TRAIN_CONFIGS:
        config = {**base_config, **lang_train_cfg}
        train_experiment(config)


if __name__ == '__main__':
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log in to wandb: {e}\nPlease run 'wandb login' in your terminal.")
    main()