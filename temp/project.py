import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'modality': 'audio',       # Options: 'audio', 'image'
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 5,               # Keep low for "1 night" demo
    'z_dim': 64,               # Latent dimension
    'c_dim': 256,              # Context dimension
    'predict_steps': 12,       # How far into future to predict (K)
    'negative_mode': 'mixed',  # Options: 'batch', 'mixed' (includes synthetic)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on {CONFIG['device']} with mode: {CONFIG['negative_mode']}")

# ==========================================
# 2. DATASETS (Mock & Structure)
# ==========================================
class MockAudioDataset(Dataset):
    """Generates synthetic waveforms for immediate testing."""
    def __init__(self, size=1000, seq_len=20480):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Synthetic sine wave with varying frequency + noise
        freq = np.random.uniform(200, 800)
        x = np.linspace(0, 10, self.seq_len)
        signal = np.sin(2 * np.pi * freq * x) + np.random.normal(0, 0.1, self.seq_len)
        # Add a "class" label (0 for low freq, 1 for high) for linear probe testing
        label = 0 if freq < 500 else 1
        return torch.FloatTensor(signal).unsqueeze(0), torch.tensor(label)

# ==========================================
# 3. ENCODERS (Multi-modal Support)
# ==========================================
class AudioEncoder(nn.Module):
    """Strided CNN to downsample high-res audio to latents."""
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, z_dim, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Conv1d(z_dim, z_dim, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Conv1d(z_dim, z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Conv1d(z_dim, z_dim, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        # Input: (N, 1, SeqLen) -> Output: (N, z_dim, LatentSteps)
        return self.net(x)

# ==========================================
# 3b. IMAGE COMPONENTS (Add this to project.py)
# ==========================================
class MockImageDataset(Dataset):
    """Generates synthetic images: Class 0 = Horizontal Bars, Class 1 = Vertical Bars"""
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create 64x64 image
        img = torch.zeros(3, 64, 64)
        label = np.random.randint(0, 2)
        
        # Simple Pattern Generation
        if label == 0: # Horizontal Bars
            for i in range(0, 64, 8):
                img[:, i:i+4, :] = 1.0 
        else: # Vertical Bars
            for i in range(0, 64, 8):
                img[:, :, i:i+4] = 1.0
                
        # Add noise to make it harder
        noise = torch.randn_like(img) * 0.2
        img = img + noise
        return img, torch.tensor(label)

class ImageEncoder(nn.Module):
    """Standard ResNet-style block for Images"""
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, z_dim, kernel_size=4, stride=2, padding=1), # 4x4
            nn.Flatten(), # Flatten 4x4 spatial grid into a sequence
            nn.Linear(z_dim * 16, z_dim) # Project to latent size
        )
    
    def forward(self, x):
        # We treat image patches as a "sequence" of length 1 for this simple demo
        # Output needs to be (N, z_dim, SequenceLength)
        z = self.net(x) # (N, z_dim)
        
        # --- FIX HERE ---
        # Increased repeat from 10 to 32. 
        # We need length > predict_steps (12) + buffer.
        return z.unsqueeze(2).repeat(1, 1, 32)

# ==========================================
# 4. CPC MODEL & NEGATIVE SAMPLING
# ==========================================
class CPCModel(nn.Module):
    def __init__(self, encoder, z_dim, c_dim, k_steps):
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.k_steps = k_steps
        
        # Autoregressive Model (Summarizes Past)
        self.gru = nn.GRU(z_dim, c_dim, batch_first=True)
        
        # Prediction Heads (W_k matrices)
        # We need a separate linear layer for each future step k
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(k_steps)
        ])

    def get_synthetic_negatives(self, z_batch):
        """
        Project Requirement: Hand-made synthetic targets.
        We add Gaussian noise to create 'hard' negatives that are 
        similar but distinct, or shuffle time.
        """
        # 1. Perturbed Negatives (Add noise)
        noise = torch.randn_like(z_batch) * 0.2
        neg_perturbed = z_batch + noise
        
        # 2. Time-Shuffled Negatives (Scramble context)
        idx = torch.randperm(z_batch.size(0))
        neg_shuffled = z_batch[idx]
        
        return neg_perturbed, neg_shuffled

    def forward(self, x):
        # 1. Encode
        z = self.encoder(x) # (N, z_dim, Time)
        z = z.permute(0, 2, 1) # (N, Time, z_dim)
        
        batch, time, dim = z.size()
        
        # 2. Randomly pick a timestep 't' to start prediction
        # We need enough history for context, and enough future for prediction
        t_sample = np.random.randint(time // 3, time - self.k_steps - 1)
        
        # 3. Generate Context (c_t) up to t_sample
        # Pass all z through GRU, then take the one at t_sample
        c_all, _ = self.gru(z)
        c_t = c_all[:, t_sample, :] # (N, c_dim)
        
        losses = []
        accuracies = []

        # 4. Contrastive Loop over K future steps
        for k in range(1, self.k_steps + 1):
            z_future_true = z[:, t_sample + k, :] # True Positive (N, z_dim)
            
            # Predict future z from current context c_t
            z_future_pred = self.predictors[k-1](c_t) # (N, z_dim)
            
            # --- InfoNCE Calculation ---
            # Similarity with Positive
            # Log-bilinear model: z^T W c
            # Here predictors(c_t) is essentially W*c
            score_pos = torch.sum(z_future_pred * z_future_true, dim=1, keepdim=True) # (N, 1)
            
            # Similarity with Negatives (Batch Negatives)
            # We treat all other samples in batch as negatives
            score_neg_batch = torch.matmul(z_future_pred, z_future_true.t()) # (N, N)
            
            # Mask out the diagonal (positive pairs) from the batch negative matrix
            mask = torch.eye(batch, device=x.device).bool()
            score_neg_batch = score_neg_batch.masked_fill(mask, float('-inf'))
            
            # --- Project Feature: Synthetic Negatives ---
            if CONFIG['negative_mode'] == 'mixed':
                # Create synthetic negatives and compute scores
                neg_pert, neg_shuf = self.get_synthetic_negatives(z_future_true)
                
                score_neg_pert = torch.sum(z_future_pred * neg_pert, dim=1, keepdim=True)
                score_neg_shuf = torch.sum(z_future_pred * neg_shuf, dim=1, keepdim=True)
                
                # Concatenate all logits: [Pos, NegBatch..., NegPert, NegShuf]
                # Note: We compute log_softmax over this dimension
                logits = torch.cat([score_pos, score_neg_batch, score_neg_pert, score_neg_shuf], dim=1)
            else:
                # Standard CPC (Batch negatives only)
                logits = torch.cat([score_pos, score_neg_batch], dim=1)
            
            # The 'label' is always index 0 (the positive score)
            labels = torch.zeros(batch, dtype=torch.long, device=x.device)
            
            # Calculate Cross Entropy Loss
            step_loss = F.cross_entropy(logits, labels)
            losses.append(step_loss)
            
            # Track accuracy (did we pick index 0?)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            accuracies.append(acc)

        return torch.stack(losses).mean(), torch.stack(accuracies).mean(), c_t

# ==========================================
# 5. CONTEXT QUALITY ANALYSIS (Metrics)
# ==========================================
def analyze_latent_space(model, loader, device):
    """
    Project Requirement: Quantify "better" context summarization.
    Calculates Alignment and Uniformity (Wang & Isola, 2020)
    referenced in your context sources (MoCoSE paper).
    """
    model.eval()
    contexts = []
    labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.encoder(x).permute(0, 2, 1)
            c, _ = model.gru(z)
            # Take last context
            contexts.append(c[:, -1, :].cpu())
            labels.append(y)
            if len(contexts) * x.size(0) > 500: break # Limit size
            
    contexts = torch.cat(contexts, dim=0)
    contexts = F.normalize(contexts, dim=1)
    
    # 1. Alignment (distance between similar items - mocked by noise perturbation here)
    # Ideally requires pairs, we simulate by adding small noise to c and measuring distance
    c_aug = contexts + torch.randn_like(contexts) * 0.1
    c_aug = F.normalize(c_aug, dim=1)
    alignment = (contexts - c_aug).norm(dim=1).pow(2).mean().item()
    
    # 2. Uniformity (how spread out they are)
    # log expectation of e^(-2 ||x-y||^2)
    pdist = torch.pdist(contexts).pow(2).mul(-2).exp().mean().log().item()
    uniformity = pdist
    
    return alignment, uniformity, contexts.numpy(), np.concatenate(labels)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
def main():
    # --- UPDATE CONFIG HERE ---
    CONFIG['modality'] = 'image'  # Change to 'image' for 2nd run
    CONFIG['z_dim'] = 64
    
    print(f"--- Running Modality: {CONFIG['modality']} ---")

    if CONFIG['modality'] == 'audio':
        dataset = MockAudioDataset(size=2000)
        encoder = AudioEncoder(CONFIG['z_dim'])
    else:
        # Load Image Data
        dataset = MockImageDataset(size=2000)
        encoder = ImageEncoder(CONFIG['z_dim'])

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # encoder = AudioEncoder(CONFIG['z_dim'])
    model = CPCModel(encoder, CONFIG['z_dim'], CONFIG['c_dim'], CONFIG['predict_steps']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    loss_history = []
    acc_history = []

    print("--- Starting Training ---")
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(CONFIG['device'])
            
            optimizer.zero_grad()
            loss, acc, _ = model(x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        avg_loss = epoch_loss / len(loader)
        avg_acc = epoch_acc / len(loader)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {avg_loss:.4f} | CPC Accuracy: {avg_acc:.4f}")

    # ==========================================
    # 7. RESULTS GENERATION (Headless Mode Fix)
    # ==========================================
    print("\n--- Generating Report Assets ---")
    
    # Set non-interactive backend
    plt.switch_backend('Agg') 
    
    # Plot Training Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Contrastive Loss')
    plt.title('CPC Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('InfoNCE Loss')
    plt.legend()
    plt.savefig('training_curve.png')  # <--- CHANGED THIS
    print("Saved training_curve.png")
    plt.close() # Clear memory

    # Metric Analysis (Project Req 3)
    align, unif, embeddings, labels = analyze_latent_space(model, loader, CONFIG['device'])
    print(f"Alignment Score (Lower is better): {align:.4f}")
    print(f"Uniformity Score (Lower is better): {unif:.4f}")

    # TSNE Visualization
    print("Visualizing Context Space...")
    tsne = TSNE(n_components=2)
    vis = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(vis[:, 0], vis[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Mock Frequency Class')
    plt.title('t-SNE of Learned Context Vectors')
    plt.savefig('tsne_plot.png')      # <--- CHANGED THIS
    print("Saved tsne_plot.png")
    plt.close()

if __name__ == "__main__":
    main()