"""
Fine-Tune Existing Speaker Encoder with New Speaker + TensorBoard Integration
Adds new speaker to your trained VCTK/RAVDESS model without losing existing knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import warnings
import time
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class SpeakerEncoder(nn.Module):
    """Same architecture as your trained model"""
    def __init__(self, input_dim=80, hidden_dim=256, embedding_dim=128, num_speakers=10):
        super(SpeakerEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x, return_embedding=False):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        embedding = self.fc_layers(x)
        
        if return_embedding:
            return embedding
        else:
            return self.classifier(embedding)


class TensorBoardMonitor:
    """TensorBoard logging for training monitoring"""
    
    def __init__(self, log_dir="runs/fine_tuning"):
        self.log_dir = Path(log_dir)
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(self.log_dir / timestamp)
        print(f"TensorBoard logs: {self.log_dir / timestamp}")
        print(f"View with: tensorboard --logdir={log_dir}")
    
    def log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Log training metrics"""
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Learning_rate', lr, epoch)
    
    def log_embeddings(self, embeddings, labels, epoch, speaker_names=None):
        """Log embedding visualizations"""
        self.writer.add_embedding(
            embeddings,
            metadata=labels,
            tag=f'speaker_embeddings_epoch_{epoch}',
            global_step=epoch
        )
    
    def log_spectrograms(self, mel_specs, epoch, tag='spectrograms'):
        """Log mel-spectrogram visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, ax in enumerate(axes.flat):
            if idx < len(mel_specs):
                mel = mel_specs[idx].cpu().numpy()
                ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f'Sample {idx+1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Mel Frequency')
        plt.tight_layout()
        self.writer.add_figure(f'{tag}/epoch_{epoch}', fig, epoch)
        plt.close(fig)
    
    def log_confusion_matrix(self, predictions, targets, epoch, speaker_names):
        """Log confusion matrix for speaker classification"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(targets, predictions)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=speaker_names, yticklabels=speaker_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure(f'confusion_matrix/epoch_{epoch}', fig, epoch)
        plt.close(fig)
    
    def log_model_graph(self, model, input_tensor):
        """Log model architecture"""
        self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


class SpeakerFineTuner:
    """Fine-tune existing speaker encoder by adding new speaker"""
    
    def __init__(self, 
                 base_model_path="best_speaker_encoder.pth",
                 new_speaker_name="custom_speaker",
                 output_dir="fine_tuned_models"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.new_speaker_name = new_speaker_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.n_mels = 80
        self.segment_length = 128
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        
        # Initialize mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=True
        ).to(self.device)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load base model
        self.base_model_path = Path(base_model_path)
        self.model = None
        self.speaker_to_id = {}
        self.id_to_speaker = {}
        self.num_speakers = 0
        self.new_speaker_id = None
        
        self._load_base_model()
    
    def _load_base_model(self):
        """Load existing trained model"""
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
        
        print(f"\nLoading base model: {self.base_model_path}")
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        # Load speaker mappings
        self.speaker_to_id = checkpoint.get('speaker_to_id', {})
        self.id_to_speaker = checkpoint.get('id_to_speaker', {})
        if isinstance(self.id_to_speaker, dict):
            self.id_to_speaker = {int(k): v for k, v in self.id_to_speaker.items()}
        
        self.num_speakers = checkpoint.get('num_speakers', len(self.speaker_to_id))
        
        print(f"Base model info:")
        print(f"  Existing speakers: {self.num_speakers}")
        print(f"  Sample speakers: {list(self.speaker_to_id.keys())[:5]}")
        
        # Add new speaker
        self.new_speaker_id = self.num_speakers
        self.speaker_to_id[self.new_speaker_name] = self.new_speaker_id
        self.id_to_speaker[self.new_speaker_id] = self.new_speaker_name
        self.num_speakers += 1
        
        print(f"\nAdding new speaker:")
        print(f"  Name: {self.new_speaker_name}")
        print(f"  ID: {self.new_speaker_id}")
        print(f"  Total speakers after fine-tuning: {self.num_speakers}")
        
        # Create model with expanded classifier
        old_num_speakers = checkpoint.get('num_speakers', len(checkpoint.get('speaker_to_id', {})))
        self.model = SpeakerEncoder(
            input_dim=80,
            hidden_dim=256,
            embedding_dim=128,
            num_speakers=self.num_speakers  # Expanded
        ).to(self.device)
        
        # Load weights with special handling for classifier layer
        state_dict = checkpoint['model_state_dict']
        
        # Load all layers except classifier
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        
        # Handle classifier layer expansion
        if 'classifier.weight' in state_dict:
            old_weight = state_dict['classifier.weight']
            old_bias = state_dict['classifier.bias']
            
            # Initialize new classifier weights
            new_weight = torch.randn(self.num_speakers, 128) * 0.01
            new_bias = torch.zeros(self.num_speakers)
            
            # Copy old weights
            new_weight[:old_num_speakers] = old_weight
            new_bias[:old_num_speakers] = old_bias
            
            model_dict['classifier.weight'] = new_weight.to(self.device)
            model_dict['classifier.bias'] = new_bias.to(self.device)
        
        self.model.load_state_dict(model_dict)
        print("\nBase model loaded and expanded successfully")
    
    def validate_audio_duration(self, audio_path):
        """Check if audio is within 30-50 seconds"""
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            return True if 30 <= duration <= 50 else False, duration
        except Exception as e:
            return False, 0
    
    def preprocess_audio(self, audio_path):
        """Preprocess single audio file"""
        try:
            waveform, orig_sr = torchaudio.load(str(audio_path))
            waveform = waveform.to(self.device)
            
            if orig_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                ).to(self.device)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            waveform = torch.clamp(waveform, -1.0, 1.0)
            
            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-6).squeeze(0)
            
            segments = self._create_segments(mel_spec)
            return segments
            
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {e}")
            return None
    
    def _create_segments(self, mel_spec):
        """Create fixed-length segments"""
        segments = []
        mel_length = mel_spec.size(1)
        
        if mel_length <= self.segment_length:
            pad_length = self.segment_length - mel_length
            padded = torch.nn.functional.pad(mel_spec, (0, pad_length), mode='reflect')
            segments.append(padded)
        else:
            num_segments = mel_length // self.segment_length
            for i in range(num_segments):
                start = i * self.segment_length
                end = start + self.segment_length
                segments.append(mel_spec[:, start:end])
            
            if mel_length % self.segment_length > 0:
                start = mel_length - self.segment_length
                segments.append(mel_spec[:, start:])
        
        return segments
    
    def prepare_training_data(self, audio_paths):
        """Prepare training data from audio files"""
        print("\nPreprocessing audio files...")
        all_segments = []
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\nProcessing audio {i}/{len(audio_paths)}: {Path(audio_path).name}")
            
            is_valid, duration = self.validate_audio_duration(audio_path)
            if not is_valid:
                print(f"  Warning: Audio duration {duration:.1f}s (should be 30-50s)")
                response = input("  Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    continue
            else:
                print(f"  Duration: {duration:.1f}s")
            
            segments = self.preprocess_audio(audio_path)
            if segments:
                all_segments.extend(segments)
                print(f"  Created {len(segments)} segments")
        
        if not all_segments:
            print("No segments created!")
            return None
        
        print(f"\nTotal segments created: {len(all_segments)}")
        return all_segments
    
    def fine_tune(self, segments, num_epochs=50, batch_size=16, 
                  freeze_backbone=False, learning_rate=0.0001):
        """Fine-tune model with new speaker data"""
        
        print(f"\n{'='*70}")
        print("FINE-TUNING CONFIGURATION")
        print(f"{'='*70}")
        print(f"  New speaker: {self.new_speaker_name}")
        print(f"  Segments: {len(segments)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Device: {self.device}")
        
        # Initialize TensorBoard
        tb_monitor = TensorBoardMonitor(log_dir="runs/fine_tuning")
        
        # Optionally freeze backbone
        if freeze_backbone:
            print("\nFreezing backbone (conv + fc layers)...")
            for param in self.model.conv_layers.parameters():
                param.requires_grad = False
            for param in self.model.fc_layers.parameters():
                param.requires_grad = False
            print("Only training classifier layer for new speaker")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        scaler = GradScaler()
        
        # Split data into train/val
        num_segments = len(segments)
        indices = list(range(num_segments))
        np.random.shuffle(indices)
        
        split_idx = int(0.8 * num_segments)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        print(f"\nData split:")
        print(f"  Training: {len(train_indices)} segments")
        print(f"  Validation: {len(val_indices)} segments")
        
        # Log model architecture (once)
        sample_input = segments[0].unsqueeze(0).to(self.device)
        tb_monitor.log_model_graph(self.model, sample_input)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        print(f"\n{'='*70}")
        print("STARTING FINE-TUNING")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            np.random.shuffle(train_indices)
            
            train_pbar = tqdm(range(0, len(train_indices), batch_size), 
                            desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            for i in train_pbar:
                batch_indices = train_indices[i:i+batch_size]
                batch_segments = torch.stack([segments[idx] for idx in batch_indices]).to(self.device)
                batch_labels = torch.full((len(batch_indices),), self.new_speaker_id, 
                                        dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(batch_segments)
                    loss = criterion(outputs, batch_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            all_embeddings = []
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                val_pbar = tqdm(range(0, len(val_indices), batch_size),
                              desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                
                for i in val_pbar:
                    batch_indices = val_indices[i:i+batch_size]
                    batch_segments = torch.stack([segments[idx] for idx in batch_indices]).to(self.device)
                    batch_labels = torch.full((len(batch_indices),), self.new_speaker_id,
                                            dtype=torch.long).to(self.device)
                    
                    with autocast():
                        outputs = self.model(batch_segments)
                        embeddings = self.model(batch_segments, return_embedding=True)
                        loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
                    
                    all_embeddings.append(embeddings.cpu())
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / (len(train_indices) / batch_size)
            avg_val_loss = val_loss / (len(val_indices) / batch_size)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            tb_monitor.log_metrics(epoch, avg_train_loss, avg_val_loss, 
                                 train_acc, val_acc, current_lr)
            
            # Log embeddings every 10 epochs
            if (epoch + 1) % 10 == 0:
                embeddings_concat = torch.cat(all_embeddings, dim=0)
                labels = [self.new_speaker_name] * len(embeddings_concat)
                tb_monitor.log_embeddings(embeddings_concat, labels, epoch)
            
            # Log spectrograms (first 4 samples)
            if (epoch + 1) % 10 == 0:
                sample_specs = [segments[i] for i in range(min(4, len(segments)))]
                tb_monitor.log_spectrograms(sample_specs, epoch, tag='new_speaker_spectrograms')
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_loss, is_best=True)
                print(f"  New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, avg_val_loss, is_best=False)
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print("FINE-TUNING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        tb_monitor.close()
        return self.model
    
    def _save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'num_speakers': self.num_speakers,
            'speaker_to_id': self.speaker_to_id,
            'id_to_speaker': self.id_to_speaker,
            'new_speaker_name': self.new_speaker_name,
            'new_speaker_id': self.new_speaker_id,
            'base_model': str(self.base_model_path)
        }
        
        checkpoint_path = self.output_dir / f"{self.new_speaker_name}_finetuned.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / f"{self.new_speaker_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path.name}")
    
    def save_model_info(self):
        """Save model metadata"""
        info = {
            'base_model': str(self.base_model_path),
            'new_speaker_name': self.new_speaker_name,
            'new_speaker_id': self.new_speaker_id,
            'total_speakers': self.num_speakers,
            'embedding_dim': 128,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'fine_tuned_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'all_speakers': list(self.speaker_to_id.keys())
        }
        
        info_path = self.output_dir / f"{self.new_speaker_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nModel info saved: {info_path}")


def main():
    """Interactive fine-tuning menu"""
    print("="*70)
    print("SPEAKER ENCODER FINE-TUNING WITH TENSORBOARD")
    print("="*70)
    print("\nThis script fine-tunes your existing VCTK/RAVDESS model")
    print("by adding a new speaker without losing existing knowledge.")
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Fine-tune model with new speaker (4 audio files)")
        print("2. Exit")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            # Check for base model
            print("\n" + "="*70)
            base_model_input = input(
                "Path to base model (default: best_speaker_encoder.pth): "
            ).strip()
            
            base_model_path = base_model_input if base_model_input else "best_speaker_encoder.pth"
            
            if not Path(base_model_path).exists():
                print(f"Error: Base model not found at {base_model_path}")
                continue
            
            # Get new speaker name
            speaker_name = input("Enter new speaker name (e.g., 'john_doe'): ").strip()
            if not speaker_name:
                print("Invalid speaker name!")
                continue
            
            # Collect audio files
            print("\nPlease provide 10 audio files (30-50 seconds each)")
            print("These should be clear recordings of the new speaker's voice")
            
            audio_paths = []
            for i in range(10):
                while True:
                    audio_path = input(f"\nAudio file {i+1}/10 path: ").strip()
                    
                    if not audio_path:
                        print("Path cannot be empty!")
                        continue
                    
                    if not Path(audio_path).exists():
                        print(f"File not found: {audio_path}")
                        retry = input("Try again? (y/n): ")
                        if retry.lower() != 'y':
                            break
                        continue
                    
                    audio_paths.append(audio_path)
                    print(f"Added: {Path(audio_path).name}")
                    break
            
            if len(audio_paths) != 10:
                print("\nInsufficient audio files. Need exactly 10 files.")
                continue
            
            # Training parameters
            print("\n" + "="*70)
            print("FINE-TUNING PARAMETERS")
            
            epochs_input = input("Number of epochs (default: 50): ").strip()
            num_epochs = int(epochs_input) if epochs_input else 50
            
            batch_input = input("Batch size (default: 16): ").strip()
            batch_size = int(batch_input) if batch_input else 16
            
            freeze_input = input("Freeze backbone layers? (y/n, default: n): ").strip().lower()
            freeze_backbone = freeze_input == 'y'
            
            lr_input = input("Learning rate (default: 0.0001): ").strip()
            learning_rate = float(lr_input) if lr_input else 0.0001
            
            # Confirm
            print("\n" + "="*70)
            print("FINE-TUNING SUMMARY:")
            print(f"  Base model: {base_model_path}")
            print(f"  New speaker: {speaker_name}")
            print(f"  Audio files: {len(audio_paths)}")
            for i, path in enumerate(audio_paths, 1):
                print(f"    {i}. {Path(path).name}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Freeze backbone: {freeze_backbone}")
            print(f"  Learning rate: {learning_rate}")
            
            confirm = input("\nStart fine-tuning? (y/n): ")
            if confirm.lower() != 'y':
                print("Fine-tuning cancelled.")
                continue
            
            # Initialize fine-tuner
            try:
                tuner = SpeakerFineTuner(
                    base_model_path=base_model_path,
                    new_speaker_name=speaker_name,
                    output_dir="fine_tuned_models"
                )
                
                # Prepare data
                segments = tuner.prepare_training_data(audio_paths)
                if segments is None:
                    print("Failed to prepare training data!")
                    continue
                
                # Fine-tune model
                model = tuner.fine_tune(
                    segments=segments,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    freeze_backbone=freeze_backbone,
                    learning_rate=learning_rate
                )
                
                # Save model info
                tuner.save_model_info()
                
                print("\n" + "="*70)
                print("FINE-TUNING COMPLETE!")
                print("="*70)
                print(f"\nModel files saved in: {tuner.output_dir}")
                print(f"  Best model: {speaker_name}_best.pth")
                print(f"  Latest checkpoint: {speaker_name}_finetuned.pth")
                print(f"  Model info: {speaker_name}_info.json")
                
                print("\nTensorBoard logs saved in: runs/fine_tuning")
                print("View with: tensorboard --logdir=runs/fine_tuning")
                
                print("\nYou can now use the fine-tuned model with:")
                print("  - Your first phase cloning script")
                print("  - Your second phase XTTS cloning script")
                print(f"\nThe model now recognizes {tuner.num_speakers} speakers total:")
                print(f"  - {tuner.num_speakers - 1} original speakers (VCTK/RAVDESS)")
                print(f"  - 1 new speaker ({speaker_name})")
                
            except Exception as e:
                print(f"\nFine-tuning failed: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == "2":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()