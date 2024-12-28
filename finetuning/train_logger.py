import os
import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class TrainingLogger:
    def __init__(self, log_dir, wandb_enabled=False):
        """
        Initialize training logger
        
        Args:
            log_dir (str): Directory to save logs
            wandb_enabled (bool): Whether to log to wandb
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f'training_log_{timestamp}.jsonl'
        
        self.wandb_enabled = wandb_enabled
        if wandb_enabled:
            import wandb
            self.wandb = wandb
    
    def log_step(self, metrics, step_type="training", master_process=True):
        """
        Log a training step
        
        Args:
            metrics (dict): Metrics to log
            step_type (str): Type of step ("training" or "eval")
            master_process (bool): Whether this is the master process
        """
        if not master_process:
            return
            
        # Add timestamp and step type
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step_type": step_type,
            **metrics
        }
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Console output
        if step_type == "eval":
            print(f"step {metrics['iter']}: train loss {metrics['train_loss']:.4f}, "
                  f"val loss {metrics['val_loss']:.4f}, lr {metrics['lr']:.2e}")
        else:
            print(f"iter {metrics['iter']}: loss {metrics['loss']:.4f}, "
                  f"time {metrics['step_time']:.2f}ms, mfu {metrics.get('mfu', 0):.2f}%")
        
        # Wandb logging
        if self.wandb_enabled:
            self.wandb.log(metrics)
    
    def plot_training_curves(self):
        """
        Create and save training curve plots
        """
        # Read the log file
        data = []
        with open(self.log_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create evaluation metrics plot
        eval_df = df[df['step_type'] == 'eval']
        
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(eval_df['iter'], eval_df['train_loss'], label='Train Loss')
        plt.plot(eval_df['iter'], eval_df['val_loss'], label='Val Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(eval_df['iter'], eval_df['lr'])
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_plots.png')
        plt.close()
    
    def get_logs_df(self):
        """
        Return logs as a pandas DataFrame
        """
        data = []
        with open(self.log_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)