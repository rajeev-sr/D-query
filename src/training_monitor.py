# src/training_monitor.py
import time
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import json

import torch

class TrainingMonitor:
    def __init__(self):
        self.training_logs = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start training monitoring"""
        self.start_time = datetime.now()
        print(f"Training started at {self.start_time}")
        
        # Check system resources
        print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("No GPU detected - training will be slower")

    def log_progress(self, step, loss, learning_rate=None):
        """Log training progress"""
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        log_entry = {
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'elapsed_time': elapsed,
            'timestamp': current_time.isoformat()
        }
        
        self.training_logs.append(log_entry)
        print(f"Step {step}: Loss={loss:.4f}, Time={elapsed:.1f}s")
    
    def save_logs(self, filepath="models/training_logs.json"):
        """Save training logs"""
        with open(filepath, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        print(f"Training logs saved to {filepath}")
    
    def plot_training_curve(self):
        """Plot training loss curve"""
        if not self.training_logs:
            return
        
        steps = [log['step'] for log in self.training_logs]
        losses = [log['loss'] for log in self.training_logs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2)
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('models/training_curve.png', dpi=150, bbox_inches='tight')
        print("Training curve saved to models/training_curve.png")