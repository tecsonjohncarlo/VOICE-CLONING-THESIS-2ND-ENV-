"""
TensorBoard Monitoring System for Voice Cloning Training
Compatible with your existing training scripts
"""

import torch
import numpy as np
import time
import psutil
import os
from pathlib import Path
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

class TensorBoardMonitor:
    """TensorBoard monitoring for voice cloning training"""
    
    def __init__(self, log_dir="runs", experiment_name=None):
        self.log_dir = Path(log_dir)
        
        if experiment_name is None:
            experiment_name = f"voice_cloning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_path = self.log_dir / experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.experiment_path))
            print(f"TensorBoard logging to: {self.experiment_path}")
            print(f"View with: tensorboard --logdir={self.log_dir}")
        else:
            self.writer = None
            print("TensorBoard not available - logging disabled")
        
        self.start_time = time.time()
        
    def log_training_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log training metrics"""
        if not self.writer:
            return
        
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Log loss difference for overfitting detection
        loss_gap = train_loss - val_loss
        acc_gap = train_acc - val_acc
        self.writer.add_scalar('Gaps/Loss_Gap', loss_gap, epoch)
        self.writer.add_scalar('Gaps/Accuracy_Gap', acc_gap, epoch)
        
    def log_system_metrics(self, epoch):
        """Log system resource usage"""
        if not self.writer:
            return
        
        # Memory usage
        memory_usage = psutil.virtual_memory().percent
        self.writer.add_scalar('System/Memory_Usage_Percent', memory_usage, epoch)
        
        # GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
            
            self.writer.add_scalar('GPU/Memory_Allocated_GB', gpu_memory_allocated, epoch)
            self.writer.add_scalar('GPU/Memory_Reserved_GB', gpu_memory_reserved, epoch)
            
            # GPU utilization (if nvidia-ml-py available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.writer.add_scalar('GPU/Utilization_Percent', gpu_util.gpu, epoch)
            except ImportError:
                pass
            except Exception:
                pass
        
        # Training time
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('System/Training_Time_Hours', elapsed_time / 3600, epoch)
    
    def log_model_weights(self, model, epoch):
        """Log model weight histograms and gradients"""
        if not self.writer:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weight histograms
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                
                # Log gradient histograms if available
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                    
                    # Log gradient norms
                    grad_norm = torch.norm(param.grad).item()
                    self.writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)
    
    def log_dataset_info(self, train_size, val_size, num_speakers, datasets):
        """Log dataset information once at the beginning"""
        if not self.writer:
            return
        
        # Log dataset sizes
        self.writer.add_text('Dataset/Info', f"""
        Training samples: {train_size:,}
        Validation samples: {val_size:,}
        Total speakers: {num_speakers:,}
        Datasets: {', '.join(datasets)}
        """)
        
        # Log as scalars for easy tracking
        self.writer.add_scalar('Dataset/Train_Size', train_size, 0)
        self.writer.add_scalar('Dataset/Val_Size', val_size, 0)
        self.writer.add_scalar('Dataset/Num_Speakers', num_speakers, 0)
    
    def log_hyperparameters(self, config):
        """Log hyperparameters"""
        if not self.writer:
            return
        
        # Convert config to strings for TensorBoard
        hparam_dict = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                hparam_dict[key] = str(value)
        
        self.writer.add_hparams(hparam_dict, {})
    
    def log_audio_sample(self, audio_tensor, sample_rate, tag, step):
        """Log audio samples"""
        if not self.writer:
            return
        
        # Ensure audio is in correct format
        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor)
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        self.writer.add_audio(tag, audio_tensor, step, sample_rate=sample_rate)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names, epoch):
        """Log confusion matrix as an image"""
        if not self.writer:
            return
        
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names[:cm.shape[1]], 
                       yticklabels=class_names[:cm.shape[0]])
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Convert plot to image
            plt.tight_layout()
            plt.savefig('temp_confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Load and log image
            from PIL import Image
            img = Image.open('temp_confusion_matrix.png')
            img_array = np.array(img)
            
            # Convert to tensor format (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            self.writer.add_image('Confusion_Matrix', img_tensor, epoch)
            
            # Cleanup
            os.remove('temp_confusion_matrix.png')
            
        except ImportError:
            print("sklearn, matplotlib, or PIL not available for confusion matrix logging")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def log_embedding_visualization(self, embeddings, labels, epoch, method='tsne'):
        """Log embedding visualization using t-SNE or UMAP"""
        if not self.writer:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            if method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
            elif method == 'umap':
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                print(f"Unknown embedding method: {method}")
                return
            
            # Reduce dimensionality
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f'Speaker Embeddings ({method.upper()}) - Epoch {epoch}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            # Save and log
            plt.tight_layout()
            plt.savefig('temp_embeddings.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Load and log image
            from PIL import Image
            img = Image.open('temp_embeddings.png')
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            self.writer.add_image(f'Embeddings/{method.upper()}', img_tensor, epoch)
            
            # Cleanup
            os.remove('temp_embeddings.png')
            
        except ImportError:
            print(f"Required libraries for {method} not available")
        except Exception as e:
            print(f"Error creating embedding visualization: {e}")
    
    def log_prosody_training_metrics(self, epoch, total_loss, reconstruction_loss, consistency_loss):
        """Log prosody adapter specific metrics"""
        if not self.writer:
            return
        
        self.writer.add_scalar('Prosody/Total_Loss', total_loss, epoch)
        self.writer.add_scalar('Prosody/Reconstruction_Loss', reconstruction_loss, epoch)
        self.writer.add_scalar('Prosody/Consistency_Loss', consistency_loss, epoch)
        
        # Log loss ratios
        if total_loss > 0:
            self.writer.add_scalar('Prosody/Reconstruction_Ratio', reconstruction_loss/total_loss, epoch)
            self.writer.add_scalar('Prosody/Consistency_Ratio', consistency_loss/total_loss, epoch)
    
    def close(self):
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()
            print("TensorBoard logging closed")


# Utility function for easy setup
def setup_tensorboard_monitoring(experiment_name=None, log_dir="runs"):
    """Setup TensorBoard monitoring with default settings"""
    return TensorBoardMonitor(log_dir=log_dir, experiment_name=experiment_name)


if __name__ == "__main__":
    # Test the monitor
    monitor = TensorBoardMonitor()
    
    # Test logging
    monitor.log_training_metrics(0, 0.5, 85.2, 0.6, 82.1, 0.001)
    monitor.log_system_metrics(0)
    
    print("TensorBoard monitor test complete")
    monitor.close()