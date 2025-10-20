"""
Debug script to diagnose model loading issues
Run this first to understand what's in your checkpoint
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path="best_speaker_encoder.pth"):
    
    print("🔍 CHECKPOINT INSPECTION")
    print("=" * 50)
    
    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print(f"📁 Current directory: {Path.cwd()}")
        print(f"📂 Files in current directory:")
        for f in Path.cwd().iterdir():
            if f.suffix == '.pth':
                print(f"   {f.name}")
        return None
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    print(f"📊 File size: {Path(checkpoint_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Load checkpoint
        print("\n📥 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("✅ Checkpoint loaded successfully!")
        print(f"\n📋 Checkpoint keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"   {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"   {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"   {key}: {type(checkpoint[key])} = {checkpoint[key]}")
        
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n🧠 Model state dict ({len(model_state)} parameters):")
            for i, (name, param) in enumerate(model_state.items()):
                print(f"   {name}: {param.shape}")
                if i >= 10:  # Show only first 10
                    print(f"   ... and {len(model_state) - 10} more parameters")
                    break
        
        # Check speaker info
        if 'speaker_to_id' in checkpoint:
            speaker_info = checkpoint['speaker_to_id']
            print(f"\n👥 Speaker information ({len(speaker_info)} speakers):")
            for i, (speaker, speaker_id) in enumerate(speaker_info.items()):
                print(f"   {speaker}: {speaker_id}")
                if i >= 5:  # Show only first 5
                    print(f"   ... and {len(speaker_info) - 5} more speakers")
                    break
        
        # Training info
        if 'epoch' in checkpoint:
            print(f"\n📈 Training info:")
            print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   Best val loss: {checkpoint.get('best_val_loss', 'Unknown')}")
            print(f"   Num speakers: {checkpoint.get('num_speakers', 'Unknown')}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Try to get more info about the error
        import traceback
        print(f"\n📋 Full error traceback:")
        traceback.print_exc()
        
        return None

def test_model_creation(checkpoint):
    """Test creating the model with checkpoint info"""
    
    print(f"\n🧠 TESTING MODEL CREATION")
    print("=" * 50)
    
    if checkpoint is None:
        print("❌ No checkpoint to test with")
        return None
    
    # Get model parameters from checkpoint
    num_speakers = checkpoint.get('num_speakers', 0)
    
    if num_speakers == 0:
        print("❌ num_speakers not found or is 0")
        return None
    
    print(f"📊 Creating model with {num_speakers} speakers...")
    
    try:
        # Import model class (same as in your training script)
        import torch.nn as nn
        
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
        
        # Create model
        model = SpeakerEncoder(
            input_dim=80,
            hidden_dim=256,
            embedding_dim=128,
            num_speakers=num_speakers
        )
        
        print("✅ Model created successfully!")
        
        # Try to load state dict
        if 'model_state_dict' in checkpoint:
            print("📥 Loading model weights...")
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model weights loaded successfully!")
            
            # Test model
            print("🧪 Testing model forward pass...")
            model.eval()
            
            # Create dummy input (batch_size=1, mel_features=80, time_steps=100)
            dummy_input = torch.randn(1, 80, 100)
            
            with torch.no_grad():
                # Test classification
                output = model(dummy_input)
                print(f"✅ Classification output shape: {output.shape}")
                
                # Test embedding extraction
                embedding = model(dummy_input, return_embedding=True)
                print(f"✅ Embedding output shape: {embedding.shape}")
            
            return model
        else:
            print("❌ model_state_dict not found in checkpoint")
            return None
            
    except Exception as e:
        print(f"❌ Error creating/loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_environment():
    """Check the environment and dependencies"""
    
    print(f"\n🔧 ENVIRONMENT CHECK")
    print("=" * 50)
    
    print(f"🐍 Python version: {torch.__version__}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💾 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"📁 Current working directory: {Path.cwd()}")
    
    # Check for required files
    required_files = [
        "best_speaker_encoder.pth",
        "training_checkpoint.pth",
        "preprocessed_data",
    ]
    
    print(f"\n📋 Required files check:")
    for file in required_files:
        path = Path(file)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024 / 1024
                print(f"   ✅ {file} ({size:.2f} MB)")
            else:
                print(f"   ✅ {file} (directory)")
        else:
            print(f"   ❌ {file} (not found)")

def main():
    """Main diagnostic function"""
    
    print("🎤 VOICE CLONING MODEL DIAGNOSTICS")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Inspect checkpoint
    checkpoint = inspect_checkpoint()
    
    # Test model creation
    model = test_model_creation(checkpoint)
    
    if model is not None:
        print(f"\n🎉 SUCCESS! Your model is working correctly.")
        print(f"🚀 You can now proceed with voice cloning!")
    else:
        print(f"\n❌ ISSUES DETECTED")
        print(f"Please fix the issues above before proceeding.")
        
        # Suggest solutions
        print(f"\n💡 SUGGESTED SOLUTIONS:")
        print(f"1. Make sure best_speaker_encoder.pth is in the current directory")
        print(f"2. Check that the file isn't corrupted (re-copy from training location)")
        print(f"3. Verify the model architecture matches your training script")
        print(f"4. Make sure PyTorch versions are compatible")

if __name__ == "__main__":
    main()