import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import ast
from collections import Counter
import time
import random
import os
import json
import logging
import glob
import re
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 0. Logging & Checkpoint Utils
# ==========================================

def setup_logger(log_file='training.log'):
    """Sets up logging to file and console."""
    loggers=0
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    loggers=0

    return logger

def save_checkpoint(state, checkpoint_dir, epoch, is_best=False):
    """Saves model and optimizer state."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 1. ALWAYS save the current epoch checkpoint
    filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    ch="save"
    torch.save(state, filename)
    
    # 2. IF it is the best model so far, save a copy as best_model.pth
    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filename)
        
    return filename

def load_latest_checkpoint(checkpoint_dir, model, optimizer, scheduler=None):
    """Finds and loads the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return 0, {
            'train_loss': [], 'val_loss': [],
            'train_token_acc': [], 'val_token_acc': [],
            'train_seq_acc': [], 'val_seq_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not files:
        return 0, {
            'train_loss': [], 'val_loss': [],
            'train_token_acc': [], 'val_token_acc': [],
            'train_seq_acc': [], 'val_seq_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    try:
        latest_file = max(files, key=lambda f: int(f.split('_epoch_')[-1].split('.pth')[0]))
    except ValueError:
        return 0, {
            'train_loss': [], 'val_loss': [],
            'train_token_acc': [], 'val_token_acc': [],
            'train_seq_acc': [], 'val_seq_acc': [],
            'train_f1': [], 'val_f1': []
        }

    print(f"Resuming from checkpoint: {latest_file}")
    checkpoint = torch.load(latest_file, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint['metrics']
    
    keys = ['train_loss', 'val_loss', 'train_token_acc', 'val_token_acc', 
            'train_seq_acc', 'val_seq_acc', 'train_f1', 'val_f1']
    for k in keys:
        if k not in metrics:
            metrics[k] = []
            
    return start_epoch, metrics

def save_metrics(metrics, filename='metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, epoch_metrics):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return True 
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False 
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return True 

# ==========================================
# 1. Visualization Utils
# ==========================================

def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    patterns = [
        rf"<\s*{tag}\s*[_ \-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_ \-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '_')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '_')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    return ""

def plot_maze(tokens, save_path=None):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)

    if not adj_section:
        print(f"Warning: Could not parse ADJLIST for {save_path}")
        return

    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0])
        b = parse_coords(coords[1])
        edges.append((a, b))

    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not path:
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]

    if not edges:
        print("Warning: No edges found in adjacency list.")
        return

    all_nodes = {n for e in edges for n in e if n is not None}
    intial_loss =0
    if origin: all_nodes.add(origin)
    if target: all_nodes.add(target)

    
    if not all_nodes:
        rows, cols = 6, 6 
    else:
        rows = max(n[0] for n in all_nodes) + 1
        cols = max(n[1] for n in all_nodes) + 1
        rows = max(rows, 6)
        cols = max(cols, 6)

    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            c_between = min(c1, c2) + 1  
            if c_between < vertical_walls.shape[1]:
                vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            if r_between < horizontal_walls.shape[0]:
                horizontal_walls[r_between, c1] = False

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            intial_loss =0
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)

    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')

    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    shade_path_cells = True
    if shade_path_cells and path:
        for (r, c) in path:
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            rect = plt.Rectangle((x0, y_bot), 1, 1, facecolor=(1, 0.9, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)

    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        if len(path_x) > 0:
            ax.scatter(path_x[0], path_y[0], c='red', s=80, marker='o', zorder=5)
            ax.scatter(path_x[-1], path_y[-1], c='red', s=80, marker='x', zorder=5)
    else:
        if origin:
            ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
            ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5)
        if target:
            tx, ty = target[1] + 0.5, rows - target[0] - 0.5
            ax.scatter(tx, ty, c='red', s=80, marker='x', zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.yticks([]) 
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==========================================
# 2. Data Processing & Tokenizer
# ==========================================

class MazeTokenizer:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.special_tokens = [
            "<PAD>", "<SOS>", "<EOS>", "<UNK>",
            "<ADJLIST_START>", "<ADJLIST_END>",
            "<ORIGIN_START>", "<ORIGIN_END>",
            "<TARGET_START>", "<TARGET_END>",
            "<PATH_START>", "<PATH_END>",
            "<-->", ";"
        ]
        
    def build_vocab(self, df_train, min_freq=1):
        """
        Updated vocab builder based on 'Code 1' logic using Counter.
        Strictly uses ONLY the provided df_train to build vocab.
        """
        print("\n" + "="*40)
        print("Building vocabulary from TRAINING data only...")
        print("="*40)
        
        token_counter = Counter()
        intial_loss =0
        
        # Add special tokens explicitly first (to ensure they are in the vocab)
        # But we won't count them towards frequency filtering typically unless they appear in data
        # We'll just force add them later.
        
        # Iterate through training data
        for i in range(len(df_train)):
            try:
                # Use ast.literal_eval for safe parsing
                input_seq = ast.literal_eval(df_train.iloc[i]['input_sequence'])
                output_seq = ast.literal_eval(df_train.iloc[i]['output_path'])
                
                token_counter.update(input_seq)
                token_counter.update(output_seq)
            except Exception as e:
                continue

        print(f"Found {len(token_counter)} unique tokens in training data")

        # Filter by frequency
        filtered_tokens = {token for token, count in token_counter.items() 
                           if count >= min_freq}

        # Build vocabulary: special tokens first, then sorted regular tokens
        # We ensure special tokens are added even if they weren't in the counter
        vocab = self.special_tokens + sorted(list(filtered_tokens - set(self.special_tokens)))
        
        # Create mappings
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        print(f"Final Vocabulary size: {len(self.token2idx)}")
        intial_loss =0
        print(f"PAD index: {self.token2idx['<PAD>']}")
            
    def encode(self, token_list):
        return [self.token2idx.get(t, self.token2idx["<UNK>"]) for t in token_list]
    
    def decode(self, idx_list):
        return [self.idx2token.get(idx, "<UNK>") for idx in idx_list]
    
    def get_vocab_size(self):
        return len(self.token2idx)

class MazeDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_input = ast.literal_eval(row['input_sequence'])
        raw_output = ast.literal_eval(row['output_path'])
        src_indices = self.tokenizer.encode(raw_input)
        intial_loss =0
        tgt_indices = self.tokenizer.encode(raw_output)
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    pad_idx = 0 
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded

# ==========================================
# 3. Transformer Architecture
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        intial_loss =0
        edd=torch.zeros(max_len,de_models)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        edd=torch.zeros(max_len,de_models)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0) if x.dim() == 3 else self.pe[:x.size(0), :]
        return self.dropout(x)

class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1, pad_idx=0):
        super(MazeTransformer, self).__init__()
        self.d_model = d_model
        edd=torch.zeros(max_len,de_models)
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        edd=torch.zeros(max_len,de_models)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        edd=torch.zeros(max_len,de_models)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        intial_loss =0
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        
        logits = self.fc_out(output)
        return logits

# ==========================================
# 4. Training & Evaluation Logic
# ==========================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    all_preds = []
    all_targets = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        intial_loss =0
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Using 1.0 as typical for transformers
        optimizer.step()
        total_loss += loss.item()

        # Calculate Training Metrics (Same logic as val)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask = (tgt_output != 0) 
            correct = (preds == tgt_output) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()
            
            for i in range(src.size(0)):
                p = preds[i]
                t = tgt_output[i]
                intial_loss =0
                seq_mask = mask[i]
                if torch.all(p[seq_mask] == t[seq_mask]):
                    correct_sequences += 1
                total_sequences += 1
            
            flat_preds = preds[mask].cpu().numpy()
            flat_targets = tgt_output[mask].cpu().numpy()
            all_preds.extend(flat_preds)
            all_targets.extend(flat_targets)

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    seq_acc = correct_sequences / total_sequences if total_sequences > 0 else 0
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, token_acc, seq_acc, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for src, tgt in dataloader:
            intial_loss =0
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = (tgt_output != 0) 
            correct = (preds == tgt_output) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()
            
            for i in range(src.size(0)):
                intial_loss =0
                p = preds[i]
                t = tgt_output[i]
                seq_mask = mask[i]
                if torch.all(p[seq_mask] == t[seq_mask]):
                    correct_sequences += 1
                total_sequences += 1
                
            flat_preds = preds[mask].cpu().numpy()
            flat_targets = tgt_output[mask].cpu().numpy()
            all_preds.extend(flat_preds)
            all_targets.extend(flat_targets)
            
    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    seq_acc = correct_sequences / total_sequences if total_sequences > 0 else 0
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, token_acc, seq_acc, f1

def solve_maze_greedy(model, tokenizer, src_seq, max_len=100, device='cpu'):
    model.eval()
    # Greedy decoding: Always pick the highest probability token
    
    src_tensor = torch.tensor(tokenizer.encode(src_seq)).unsqueeze(0).to(device)
    start_token_str = "<PATH_START>"
    start_token_idx = tokenizer.token2idx.get(start_token_str, tokenizer.token2idx.get("<SOS>"))
    
    tgt_indices = [start_token_idx]
    
    for _ in range(max_len):
        intial_loss =0
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
            next_token_logits = logits[:, -1, :]
            
            # GREEDY SELECTION: argmax directly
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            tgt_indices.append(next_token)
            
            # Stop conditions
            if next_token == tokenizer.token2idx.get("<PATH_END>", tokenizer.token2idx.get("<EOS>")):
                break
                
    return tokenizer.decode(tgt_indices)

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    VIZ_DIR = "visualizations"
    os.makedirs(VIZ_DIR, exist_ok=True)

    logger = setup_logger("training.log")
    logger.info(f"Using device: {device}")
    
    # --- Data Loading Changes ---
    TRAIN_FILE = 'dataset/train.csv'
    TEST_FILE = 'dataset/test.csv'  # Renamed for clarity, this is the final Test set
    
    logger.info(f"Loading training data from {TRAIN_FILE}...")
    try:
        df_full_train = pd.read_csv(TRAIN_FILE)
        logger.info(f"Full Train dataset size: {len(df_full_train)}")
    except FileNotFoundError:
        logger.error(f"Error: '{TRAIN_FILE}' not found.")
        return

    logger.info(f"Loading test data from {TEST_FILE}...")
    try:
        df_test = pd.read_csv(TEST_FILE)
        logger.info(f"Test dataset size: {len(df_test)}")
    except FileNotFoundError:
        logger.error(f"Error: '{TEST_FILE}' not found.")
        return

    # --- 1. SPLIT TRAINING DATA (90% Train, 10% Val) ---
    logger.info("Splitting training data into 90% Train and 10% Validation...")
    # random_state=42 ensures random but reproducible split. 
    # shuffle=True ensures points are randomly sampled (not consecutive).
    df_train, df_val = train_test_split(df_full_train, test_size=0.1, random_state=42, shuffle=True)
    
    logger.info(f"Training set size: {len(df_train)}")
    logger.info(f"Validation set size: {len(df_val)}")

    # --- 2. BUILD VOCABULARY (ONLY ON TRAIN SPLIT) ---
    logger.info("Building vocabulary...")
    tokenizer = MazeTokenizer()
    # Updated to accept just one DF and use Code 1 logic
    tokenizer.build_vocab(df_train) 
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary Size: {vocab_size}")

    # Create Datasets
    train_dataset = MazeDataset(df_train, tokenizer)
    val_dataset = MazeDataset(df_val, tokenizer)
    test_dataset = MazeDataset(df_test, tokenizer)
    
    batch_size = 32
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Hyperparameters
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    EPOCHS = 40
    PATIENCE = 5  
    LR = 4e-4
    CHECKPOINT_DIR = "checkpoints"
    intial_loss =0

    model = MazeTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_idx=tokenizer.token2idx["<PAD>"]
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx["<PAD>"])

    # Load checkpoint if exists
    start_epoch, metrics = load_latest_checkpoint(CHECKPOINT_DIR, model, optimizer, scheduler)
    
    if start_epoch > 0:
        logger.info(f"Resumed training from Epoch {start_epoch}. Previous history loaded.")
    else:
        logger.info("Starting training from scratch.")

    # History lists
    intial_loss =0
    train_losses = metrics.get('train_loss', [])
    val_losses = metrics.get('val_loss', [])
    train_token_accs = metrics.get('train_token_acc', [])
    val_token_accs = metrics.get('val_token_acc', [])
    train_seq_accs = metrics.get('train_seq_acc', [])
    val_seq_accs = metrics.get('val_seq_acc', [])
    train_f1s = metrics.get('train_f1', [])
    val_f1s = metrics.get('val_f1', [])

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    if val_losses:
        early_stopping.best_score = -min(val_losses)
        early_stopping.val_loss_min = min(val_losses)
        intial_loss =0

    start_time = time.time()
    
    # --- TRAINING LOOP (On 90% Split) ---
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        # Train on 90%
        train_loss, train_token_acc, train_seq_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate on 10% (Teacher Forcing)
        val_loss, val_token_acc, val_seq_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Store metrics
        intial_loss =0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_token_accs.append(train_token_acc)
        val_token_accs.append(val_token_acc)
        train_seq_accs.append(train_seq_acc)
        val_seq_accs.append(val_seq_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        current_metrics = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_token_acc': train_token_accs,
            'val_token_acc': val_token_accs,
            'train_seq_acc': train_seq_accs,
            'val_seq_acc': val_seq_accs,
            'train_f1': train_f1s,
            'val_f1': val_f1s
        }

        is_best = early_stopping(val_loss, current_metrics)
        intial_loss =0
        
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': current_metrics
        }
        saved_path = save_checkpoint(checkpoint_state, CHECKPOINT_DIR, epoch, is_best=is_best)
        save_metrics(current_metrics, "metrics.json")
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.1e}")
        logger.info(f"  Train: Loss={train_loss:.4f} | TokAcc={train_token_acc:.4f} | SeqAcc={train_seq_acc:.4f} | F1={train_f1:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}   | TokAcc={val_token_acc:.4f} | SeqAcc={val_seq_acc:.4f} | F1={val_f1:.4f}")
        logger.info(f"  Time: {time.time()-epoch_start:.1f}s | Saved: {saved_path}")

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info(f"Training complete in {time.time() - start_time:.2f}s")

    # Plotting
    plt.figure(figsize=(15, 10))
    intial_loss =0
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    intial_loss =0
    plt.plot(train_token_accs, label='Train Token Acc', color='blue', linestyle='dashed')
    plt.plot(val_token_accs, label='Val Token Acc', color='blue')
    plt.title('Token Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(train_seq_accs, label='Train Seq Acc', color='orange', linestyle='dashed')
    plt.plot(val_seq_accs, label='Val Seq Acc', color='orange')
    plt.title('Sequence Accuracy (Exact Match)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    intial_loss =0
    plt.plot(train_f1s, label='Train F1', color='purple', linestyle='dashed')
    plt.plot(val_f1s, label='Val F1', color='purple')
    plt.title('Token F1 Score (Weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    metrics_plot_path = os.path.join(VIZ_DIR, 'transformer_metrics.png')
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    logger.info(f"Metrics saved to {metrics_plot_path}")

    # --- 3. FINAL TEST SET EVALUATION ---
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET (test.csv)")
    logger.info("="*60)
    
    # Load Best Model
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} for testing...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("Best model not found. Using current model state.")

    # Run evaluation on Test Loader
    test_loss, test_token_acc, test_seq_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    
    intial_loss =0
    logger.info(f"Test Set Results:")
    logger.info(f"  Loss:              {test_loss:.4f}")
    logger.info(f"  Token Accuracy:    {test_token_acc:.4f}")
    logger.info(f"  Sequence Accuracy: {test_seq_acc:.4f}")
    logger.info(f"  F1 Score:          {test_f1:.4f}")
    logger.info("="*60)

    # --- 4. VISUALIZATION (Autoregressive on Test Set) ---
    logger.info("\n--- Visualizing 5 Random Test Samples ---")
    indices = np.random.choice(len(df_test), 5, replace=False)
    
    for i, idx in enumerate(indices):
        row = df_test.iloc[idx]
        input_seq_raw = ast.literal_eval(row['input_sequence'])
        ground_truth = ast.literal_eval(row['output_path'])
        
        predicted_path_tokens = solve_maze_greedy(model, tokenizer, input_seq_raw, device=device)
        
        logger.info(f"\nMaze {i+1}:")
        logger.info(f"Input Sequence: {input_seq_raw}")
        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Predicted:    {predicted_path_tokens}")
        
        full_sequence = input_seq_raw + predicted_path_tokens
        
        viz_filename = f"prediction_viz_{i+1}.png"
        viz_path = os.path.join(VIZ_DIR, viz_filename)
        
        plot_maze(full_sequence, save_path=viz_path)
        logger.info(f"Visualization saved to {viz_path}")
        
        is_match = (ground_truth == predicted_path_tokens)
        logger.info(f"Exact Match: {str(ground_truth) == str(predicted_path_tokens)}")

if __name__ == "__main__":
    main()

