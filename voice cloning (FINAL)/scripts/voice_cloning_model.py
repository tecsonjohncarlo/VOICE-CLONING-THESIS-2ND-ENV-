"""
Advanced Voice Cloning Model Trainer
Uses pre-trained speaker encoder as feature extractor for voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import gc
import time
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard matplotlib")

class TensorBoardMonitor:
    """Enhanced TensorBoard monitoring for voice cloning training"""
    
    def __init__(self, log_dir="runs/voice_cloning"):
        if not TENSORBOARD_AVAILABLE:
            print("TensorBoard not available - metrics will not be logged")
            self.writer = None
            return
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique run directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.log_dir / f"run_{timestamp}"
        
        self.writer = SummaryWriter(str(run_dir))
        print(f"TensorBoard logging to: {run_dir}")
        print(f"View with: tensorboard --logdir {self.log_dir}")
    
    def log_training_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate):
        if self.writer is None:
            return
            
        self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'Train': train_acc,
            'Validation': val_acc
        }, epoch)
        
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Log loss difference for overfitting detection
        loss_diff = abs(train_loss - val_loss)
        self.writer.add_scalar('Overfitting/Loss_Difference', loss_diff, epoch)
        
        acc_diff = abs(train_acc - val_acc)
        self.writer.add_scalar('Overfitting/Accuracy_Difference', acc_diff, epoch)
    
    def log_voice_cloning_metrics(self, epoch, reconstruction_loss, adversarial_loss, perceptual_loss, total_loss):
        if self.writer is None:
            return
            
        self.writer.add_scalars('Voice_Cloning_Loss', {
            'Reconstruction': reconstruction_loss,
            'Adversarial': adversarial_loss,
            'Perceptual': perceptual_loss,
            'Total': total_loss
        }, epoch)
    
    def log_audio_sample(self, tag, audio_tensor, sample_rate, epoch):
        if self.writer is None:
            return
            
        # Ensure audio is in correct format for tensorboard
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        self.writer.add_audio(tag, audio_tensor, epoch, sample_rate=sample_rate)
    
    def log_spectrogram(self, tag, spectrogram, epoch):
        if self.writer is None:
            return
            
        # Convert spectrogram to image
        if isinstance(spectrogram, torch.Tensor):
            spec_np = spectrogram.detach().cpu().numpy()
        else:
            spec_np = spectrogram
            
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{tag}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel Frequency')
        plt.colorbar(im, ax=ax)
        
        self.writer.add_figure(f'{tag}_Spectrogram', fig, epoch)
        plt.close(fig)
    
    def log_model_weights(self, model, epoch):
        if self.writer is None:
            return
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
    
    def close(self):
        if self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed")

# Original Speaker Encoder (from your training script)
class SpeakerEncoder(nn.Module):
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

class VoiceCloningGenerator(nn.Module):
    """Voice Cloning Generator that takes text features and speaker embedding"""
    
    def __init__(self, 
                 speaker_embedding_dim=128,
                 text_embedding_dim=256,
                 hidden_dim=512,
                 n_mels=80,
                 max_seq_len=1000):
        super(VoiceCloningGenerator, self).__init__()
        
        self.speaker_embedding_dim = speaker_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.max_seq_len = max_seq_len
        
        # Text encoder (simplified - in practice you'd use something like Tacotron2's encoder)
        self.text_encoder = nn.Sequential(
            nn.Embedding(256, text_embedding_dim),  # Character-level
            nn.LSTM(text_embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True),
        )
        
        # Speaker conditioning
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder (mel-spectrogram generation)
        self.decoder_rnn = nn.LSTM(
            input_size=hidden_dim + n_mels,  # Previous mel + context
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Output projection
        self.mel_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_mels)
        )
        
        # Stop token prediction
        self.stop_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, text_input, speaker_embedding, target_mel=None, max_len=None):
        batch_size = text_input.size(0)
        
        # Encode text
        text_embedded = self.text_encoder.forward(text_input)
        if isinstance(text_embedded, tuple):
            text_encoded, _ = text_embedded
        else:
            text_encoded = text_embedded
            
        # Project speaker embedding
        speaker_context = self.speaker_proj(speaker_embedding)  # [batch, hidden_dim]
        speaker_context = speaker_context.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Repeat speaker context to match text length
        text_len = text_encoded.size(1)
        speaker_context = speaker_context.expand(-1, text_len, -1)  # [batch, text_len, hidden_dim]
        
        # Combine text and speaker information
        encoder_output = text_encoded + speaker_context
        
        # Decoder
        if target_mel is not None:
            # Training mode - teacher forcing
            decoder_input = torch.zeros(batch_size, 1, self.n_mels, device=text_input.device)
            decoder_input = torch.cat([decoder_input, target_mel[:, :-1]], dim=1)
            
            mel_outputs = []
            stop_outputs = []
            
            hidden = None
            for t in range(target_mel.size(1)):
                # Attention over encoder output
                query = encoder_output[:, [t % text_len]]  # Simplified attention
                
                # RNN step
                rnn_input = torch.cat([decoder_input[:, [t]], query], dim=-1)
                rnn_output, hidden = self.decoder_rnn(rnn_input, hidden)
                
                # Generate mel and stop token
                mel_output = self.mel_proj(rnn_output)
                stop_output = self.stop_proj(rnn_output)
                
                mel_outputs.append(mel_output)
                stop_outputs.append(stop_output)
            
            mel_outputs = torch.cat(mel_outputs, dim=1)
            stop_outputs = torch.cat(stop_outputs, dim=1)
            
            return mel_outputs, stop_outputs
            
        else:
            # Inference mode
            max_len = max_len or self.max_seq_len
            
            mel_outputs = []
            stop_outputs = []
            
            decoder_input = torch.zeros(batch_size, 1, self.n_mels, device=text_input.device)
            hidden = None
            
            for t in range(max_len):
                # Attention over encoder output
                query = encoder_output[:, [t % text_len]]
                
                # RNN step
                rnn_input = torch.cat([decoder_input, query], dim=-1)
                rnn_output, hidden = self.decoder_rnn(rnn_input, hidden)
                
                # Generate mel and stop token
                mel_output = self.mel_proj(rnn_output)
                stop_output = self.stop_proj(rnn_output)
                
                mel_outputs.append(mel_output)
                stop_outputs.append(stop_output)
                
                # Update decoder input for next step
                decoder_input = mel_output
                
                # Check stop condition
                if torch.sigmoid(stop_output).max() > 0.5:
                    break
            
            mel_outputs = torch.cat(mel_outputs, dim=1)
            stop_outputs = torch.cat(stop_outputs, dim=1)
            
            return mel_outputs, stop_outputs

class VoiceCloningDataset(Dataset):
    """Dataset for voice cloning training"""
    
    def __init__(self, data_dir, speaker_encoder, device, max_text_len=100):
        self.data_dir = Path(data_dir)
        self.speaker_encoder = speaker_encoder
        self.device = device
        self.max_text_len = max_text_len
        
        # Load data files
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} training samples")
    
    def _load_samples(self):
        """Load training samples - expects audio files with corresponding text"""
        samples = []
        
        # Look for paired audio-text files
        audio_files = list(self.data_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Look for corresponding text file
            text_file = audio_file.with_suffix('.txt')
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                samples.append({
                    'audio_path': str(audio_file),
                    'text': text
                })
        
        return samples
    
    def _text_to_sequence(self, text):
        """Convert text to character sequence"""
        # Simple character-level encoding
        chars = list(text.lower())
        sequence = [ord(c) for c in chars if ord(c) < 256]
        
        # Pad or truncate
        if len(sequence) > self.max_text_len:
            sequence = sequence[:self.max_text_len]
        else:
            sequence.extend([0] * (self.max_text_len - len(sequence)))
        
        return torch.LongTensor(sequence)
    
    def _extract_mel_spectrogram(self, audio_path):
        """Extract mel spectrogram matching your training preprocessing"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Extract mel spectrogram (match your preprocessing exactly)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=80,
                fmin=0,
                fmax=sr//2
            )
            
            # Convert to log scale and normalize (match your preprocessing)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return torch.FloatTensor(mel_spec_norm)
            
        except Exception as e:
            print(f"Error extracting mel spectrogram from {audio_path}: {e}")
            return None
    
    def _get_speaker_embedding(self, mel_spec):
        """Extract speaker embedding using pre-trained encoder"""
        try:
            with torch.no_grad():
                # Ensure correct input shape
                if mel_spec.dim() == 2:
                    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
                
                mel_spec = mel_spec.to(self.device)
                
                # Get speaker embedding
                embedding = self.speaker_encoder(mel_spec, return_embedding=True)
                return embedding.squeeze(0).cpu()  # Remove batch dim and move to CPU
                
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract mel spectrogram
        mel_spec = self._extract_mel_spectrogram(sample['audio_path'])
        if mel_spec is None:
            return None
        
        # Get speaker embedding
        speaker_embedding = self._get_speaker_embedding(mel_spec)
        if speaker_embedding is None:
            return None
        
        # Process text
        text_sequence = self._text_to_sequence(sample['text'])
        
        return {
            'text': text_sequence,
            'mel_spec': mel_spec.T,  # Transpose to [time, mels]
            'speaker_embedding': speaker_embedding,
            'audio_path': sample['audio_path']
        }

def collate_fn(batch):
    """Collate function to handle variable length sequences"""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Get maximum lengths
    max_mel_len = max(item['mel_spec'].size(0) for item in batch)
    
    # Pad sequences
    texts = torch.stack([item['text'] for item in batch])
    speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])
    
    # Pad mel spectrograms
    mel_specs = []
    for item in batch:
        mel = item['mel_spec']
        if mel.size(0) < max_mel_len:
            pad_len = max_mel_len - mel.size(0)
            padded_mel = torch.cat([mel, torch.zeros(pad_len, mel.size(1))], dim=0)
        else:
            padded_mel = mel[:max_mel_len]
        mel_specs.append(padded_mel)
    
    mel_specs = torch.stack(mel_specs)
    
    return {
        'text': texts,
        'mel_spec': mel_specs,
        'speaker_embedding': speaker_embeddings,
        'mel_lengths': torch.LongTensor([item['mel_spec'].size(0) for item in batch])
    }

class VoiceCloningTrainer:
    """Main trainer for voice cloning model"""
    
    def __init__(self,
                 speaker_encoder_path,
                 data_dir,
                 output_dir="voice_cloning_models",
                 device=None,
                 batch_size=16):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing Voice Cloning Trainer on {self.device}")
        
        # Load pre-trained speaker encoder
        print("Loading pre-trained speaker encoder...")
        self.speaker_encoder = self._load_speaker_encoder(speaker_encoder_path)
        
        if self.speaker_encoder is None:
            raise ValueError("Failed to load speaker encoder")
        
        # Create dataset
        print("Creating dataset...")
        self.dataset = VoiceCloningDataset(data_dir, self.speaker_encoder, self.device)
        
        if len(self.dataset) == 0:
            raise ValueError("No training data found. Ensure audio-text pairs exist in data directory")
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        # Create voice cloning model
        print("Creating voice cloning model...")
        speaker_embedding_dim = self.speaker_encoder.fc_layers[3].out_features
        
        self.generator = VoiceCloningGenerator(
            speaker_embedding_dim=speaker_embedding_dim,
            text_embedding_dim=256,
            hidden_dim=512,
            n_mels=80
        ).to(self.device)
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.stop_loss = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        # TensorBoard monitor
        self.monitor = TensorBoardMonitor()
        
        print("Voice Cloning Trainer initialized successfully!")
    
    def _load_speaker_encoder(self, checkpoint_path):
        """Load pre-trained speaker encoder"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            num_speakers = checkpoint.get('num_speakers', 0)
            if num_speakers == 0:
                print("Error: num_speakers not found in checkpoint")
                return None
            
            # Create speaker encoder
            speaker_encoder = SpeakerEncoder(
                input_dim=80,
                hidden_dim=256,
                embedding_dim=128,
                num_speakers=num_speakers
            ).to(self.device)
            
            # Load weights
            speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
            speaker_encoder.eval()
            
            # Freeze speaker encoder
            for param in speaker_encoder.parameters():
                param.requires_grad = False
            
            print(f"Speaker encoder loaded: {num_speakers} speakers")
            return speaker_encoder
            
        except Exception as e:
            print(f"Error loading speaker encoder: {e}")
            return None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            if batch is None:
                continue
            
            # Move to device
            text = batch['text'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            speaker_embedding = batch['speaker_embedding'].to(self.device)
            mel_lengths = batch['mel_lengths']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_mel, pred_stop = self.generator(
                text, speaker_embedding, target_mel=mel_spec
            )
            
            # Calculate losses
            mel_loss = self.reconstruction_loss(pred_mel, mel_spec)
            
            # Create stop targets (1 at end of sequence, 0 elsewhere)
            stop_targets = torch.zeros_like(pred_stop)
            for i, length in enumerate(mel_lengths):
                if length < pred_stop.size(1):
                    stop_targets[i, length-1] = 1.0
            
            stop_loss = self.stop_loss(pred_stop.squeeze(-1), stop_targets)
            
            total_loss_batch = mel_loss + 0.1 * stop_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Mel Loss': f'{mel_loss.item():.4f}',
                'Stop Loss': f'{stop_loss.item():.4f}',
                'Total': f'{total_loss_batch.item():.4f}'
            })
            
            # Log to TensorBoard
            if num_batches % 50 == 0:
                self.monitor.log_voice_cloning_metrics(
                    self.epoch * len(self.dataloader) + num_batches,
                    mel_loss.item(),
                    0,  # adversarial_loss
                    0,  # perceptual_loss
                    total_loss_batch.item()
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'speaker_encoder_dim': self.generator.speaker_embedding_dim
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_voice_cloning_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def train(self, num_epochs=100):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            avg_loss = self.train_epoch()
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for improvement
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
            
            # Log epoch metrics
            self.monitor.log_training_metrics(
                epoch, avg_loss, 0, avg_loss, 0, current_lr
            )
        
        print("Training completed!")
        self.monitor.close()

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Voice Cloning Model')
    parser.add_argument('--speaker_encoder', type=str, required=True,
                      help='Path to pre-trained speaker encoder (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing paired audio-text files')
    parser.add_argument('--output_dir', type=str, default='voice_cloning_models',
                      help='Output directory for trained models')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--device', type=str, default=None,
                      help='Training device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = VoiceCloningTrainer(
        speaker_encoder_path=args.speaker_encoder,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs)

if __name__ == "__main__":
    main()