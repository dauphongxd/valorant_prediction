# test.py (Corrected)
import torch  # <-- ADD THIS LINE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# You can add this for an even better test
if device == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")