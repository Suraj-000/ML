import sys
import torch
import pandas as pd
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import os 
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers=1,pad_idx=0,dropout=0.0):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.pad_idx=pad_idx

        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.rnn=nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers>1 else 0.0
        )

    def forward(self, input,input_lens):

        batch_size=input.size(0)
        data_embed=self.embedding(input)

        dev=input.device
        packed = nn.utils.rnn.pack_padded_sequence(data_embed, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self,encode_hidden_dim,decode_hidden_dim,attention_hidden_dim):
        super().__init__()
        self.w_dec=nn.Linear(decode_hidden_dim,attention_hidden_dim,bias=False)
        self.w_enc=nn.Linear(encode_hidden_dim,attention_hidden_dim,bias=False)
        self.v=nn.Linear(attention_hidden_dim,1,bias=False)

    def forward(self,dec_hidden,enc_outputs,mask=None):
        dec_exp=dec_hidden.unsqueeze(1)
        score=self.v(torch.tanh(self.w_dec(dec_exp)+self.w_enc(enc_outputs))).squeeze(-1)
        if mask is not None:score=score.masked_fill(mask==0,-1e9)
        attention_weights=torch.softmax(score,dim=1)
        context=torch.bmm(attention_weights.unsqueeze(1),enc_outputs).squeeze(1)
        return attention_weights,context
    
class DecoderRNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,encoder_hidden_dim,decoder_hidden_dim,num_layers=1,pad_idx=0,attention_hidden=256,dropout=0.0):
        super().__init__()
        self.vocab_size=vocab_size
        self.decoder_hidden_dim=decoder_hidden_dim
        self.num_layers=num_layers
        self.pad_idx=pad_idx

        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)

        self.rnn=nn.RNN(
            input_size=embedding_dim+encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers>1 else 0.0             
        )
        self.attention=BahdanauAttention(encode_hidden_dim=encoder_hidden_dim,decode_hidden_dim=decoder_hidden_dim,attention_hidden_dim=attention_hidden)
        self.out=nn.Linear(decoder_hidden_dim+encoder_hidden_dim+embedding_dim,vocab_size)

    def create_mask(self,encode_inputs,pad_idx):
        return (encode_inputs!=pad_idx).long()

    def forward(self,previous_token,previous_hidden,encoder_outputs,encoder_mask):
        embed=self.embedding(previous_token).unsqueeze(1)
        decode_top=previous_hidden[-1]
        attention_weights,context=self.attention(decode_top,encoder_outputs,mask=encoder_mask)
        rnn_input=torch.cat([embed,context.unsqueeze(1)],dim=-1)
        rnn_out,hidden=self.rnn(rnn_input,previous_hidden )
        rnn_out=rnn_out.squeeze(1)

        out_cat=torch.cat([rnn_out,context,embed.squeeze(1)],dim=-1)
        logits=self.out(out_cat)
        return logits,hidden,attention_weights


def get_rnn_vocab():
    """
    Returns the specific vocab used for the RNN model (49 tokens).
    """
    vocab = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<ADJLIST_START>': 3, '(3,5)': 4, '<-->': 5, '(2,5)': 6, 
        ';': 7, '(3,2)': 8, '(3,3)': 9, '(3,4)': 10, '(0,4)': 11, '(1,4)': 12, '(2,3)': 13, '(2,4)': 14, 
        '(1,5)': 15, '(0,1)': 16, '(0,0)': 17, '(0,2)': 18, '(2,2)': 19, '(1,3)': 20, '(1,0)': 21, 
        '(1,2)': 22, '(0,3)': 23, '<ADJLIST_END>': 24, '<ORIGIN_START>': 25, '<ORIGIN_END>': 26, 
        '<TARGET_START>': 27, '<TARGET_END>': 28, '<PATH_START>': 29, '(5,4)': 30, '(5,3)': 31, 
        '(5,2)': 32, '(3,1)': 33, '(3,0)': 34, '(4,0)': 35, '(5,0)': 36, '(4,2)': 37, '(4,1)': 38, 
        '(5,5)': 39, '(4,5)': 40, '(2,1)': 41, '(2,0)': 42, '(5,1)': 43, '(0,5)': 44, '(4,4)': 45, 
        '(4,3)': 46, '(1,1)': 47, '<PATH_END>': 48
    }
    return vocab

def get_transformer_vocab():
    """
    Returns the exact vocabulary provided by the user for the Transformer (50 tokens).
    """
    vocab = {
        "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, 
        "<ADJLIST_START>": 4, "<ADJLIST_END>": 5, 
        "<ORIGIN_START>": 6, "<ORIGIN_END>": 7, 
        "<TARGET_START>": 8, "<TARGET_END>": 9, 
        "<PATH_START>": 10, "<PATH_END>": 11, 
        "<-->": 12, ";": 13, 
        "(0,0)": 14, "(0,1)": 15, "(0,2)": 16, "(0,3)": 17, "(0,4)": 18, "(0,5)": 19, 
        "(1,0)": 20, "(1,1)": 21, "(1,2)": 22, "(1,3)": 23, "(1,4)": 24, "(1,5)": 25, 
        "(2,0)": 26, "(2,1)": 27, "(2,2)": 28, "(2,3)": 29, "(2,4)": 30, "(2,5)": 31, 
        "(3,0)": 32, "(3,1)": 33, "(3,2)": 34, "(3,3)": 35, "(3,4)": 36, "(3,5)": 37, 
        "(4,0)": 38, "(4,1)": 39, "(4,2)": 40, "(4,3)": 41, "(4,4)": 42, "(4,5)": 43, 
        "(5,0)": 44, "(5,1)": 45, "(5,2)": 46, "(5,3)": 47, "(5,4)": 48, "(5,5)": 49
    }
    return vocab

def safe_eval(x):
    if isinstance(x, str): return eval(x)
    return x

def load_data(path, model_type):
    data = pd.read_csv(path)
    input_sequence = data['input_sequence'].apply(safe_eval).tolist()
    
    # Check if 'output_path' or 'output_sequence' exists, otherwise use empty lists
    if 'output_path' in data.columns:
        output_sequence = data['output_path'].apply(safe_eval).tolist()
    else:
        output_sequence = [[] for _ in range(len(input_sequence))]
    
    # SELECT VOCAB BASED ON MODEL TYPE
    if model_type == 'transformer':
        vocab = get_transformer_vocab()
        unk_idx = vocab['<UNK>']
    else:
        vocab = get_rnn_vocab()
        unk_idx = 0 # RNN vocab uses 0 (PAD) as fallback usually if UNK is missing, or we handle it manually
        
    input_data = [[vocab.get(token, unk_idx) for token in seq] for seq in input_sequence]
    
    return input_data, output_sequence, vocab


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0) if x.dim() == 3 else self.pe[:x.size(0), :]
        return self.dropout(x)

class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1, pad_idx=0):
        super(MazeTransformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        
        logits = self.fc_out(output)
        return logits

# ==========================================
# Evaluation Functions
# ==========================================

def load_rnn_model(model_path, vocab):
    weights = torch.load(model_path, map_location=device)
    encoder = EncoderRNN(len(vocab), 128, 512, 2, 0, 0.0)
    decoder = DecoderRNN(len(vocab), 128, 512, 512, 2, 0, 256, 0)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.load_state_dict(weights['encoder'])
    decoder.load_state_dict(weights['decoder'])
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def eval_rnn(model_path, input, output, output_path, vocab):
    encoder, decoder = load_rnn_model(model_path, vocab)
    all_pred_texts = []
    
    print("Evaluating RNN...")
    for idx in range(len(input)):
        input_seq = input[idx]
        if not input_seq: 
            all_pred_texts.append("[]")
            continue

        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        input_len = torch.tensor([len(input_seq)], dtype=torch.long).to(device)

        with torch.no_grad():
            encoder_outputs, encoder_hidden = encoder(input_tensor, input_len)
            encoder_mask = (input_tensor != 0).to(device)

            decoder_init_hidden = encoder_hidden
            if decoder_init_hidden.size(0) != decoder.num_layers:
                decoder_init_hidden = decoder_init_hidden[:decoder.num_layers]

            previous_token = torch.full((1,), 1, dtype=torch.long, device=device) # SOS

            preds = []
            max_output_len = 50
            for t in range(max_output_len):
                logits, decoder_init_hidden, attention = decoder.forward(previous_token, decoder_init_hidden, encoder_outputs, encoder_mask)
                pred_token = torch.argmax(logits, dim=-1)
                preds.append(pred_token.item())
                previous_token = pred_token
                if pred_token.item() == 2: break # EOS
        
        inv_vocab = {v: k for k, v in vocab.items()}
        pred_tokens = [inv_vocab.get(i, '<UNK>') for i in preds if i != 0]
        pred_clean = [tok for tok in pred_tokens if tok not in ["<EOS>"]]
        
        pred_text_for_csv = "[" + ", ".join(f"'{tok}'" for tok in pred_clean) + "]"
        all_pred_texts.append(pred_text_for_csv)
        
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx} samples...")

    # Save
    orig_df = pd.read_csv(sys.argv[3])
    orig_df['output_path'] = all_pred_texts
    orig_df.to_csv(output_path, index=False)
    print(f"\nSaved predicted paths to: {output_path}")

def load_transformer_model(model_path, vocab):
    # Fixed hyperparameters matching training
    model = MazeTransformer(
        vocab_size=len(vocab),
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=vocab.get('<PAD>', 0)
    ).to(device)
    
    print(f"Loading Transformer weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def eval_transformer(model_path, input, output, output_path, vocab):
    model = load_transformer_model(model_path, vocab)
    all_pred_texts = []
    
    inv_vocab = {v: k for k, v in vocab.items()}
    start_token_idx = vocab.get('<PATH_START>', vocab.get('<SOS>', 1))
    end_token_idx = vocab.get('<PATH_END>', vocab.get('<EOS>', 2))
    
    print(f"Evaluating Transformer (Vocab size: {len(vocab)})...")
    
    for idx in range(len(input)):
        input_seq = input[idx]
        if not input_seq: 
            all_pred_texts.append("[]")
            continue

        src_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        # Greedy Decoding
        tgt_indices = [start_token_idx]
        max_len = 60 
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(src_tensor, tgt_tensor)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                
                tgt_indices.append(next_token)
                
                if next_token == end_token_idx:
                    break
        
        # Decode
        generated_path = tgt_indices[1:] # Skip start token
        pred_tokens = [inv_vocab.get(i, '<UNK>') for i in generated_path]
        
        # Clean tokens
        pred_clean = [tok for tok in pred_tokens if tok not in ["<EOS>", "<PATH_END>", "<PAD>"]]
        
        pred_text_for_csv = "[" + ", ".join(f"'{tok}'" for tok in pred_clean) + "]"
        all_pred_texts.append(pred_text_for_csv)
        
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx} samples...")

    # Save
    try:
        orig_df = pd.read_csv(sys.argv[3])
        orig_df['predicted_path'] = all_pred_texts
        orig_df.to_csv(output_path, index=False)
        print(f"\nSaved predicted paths to: {output_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    
    print(f"DEBUG: Script started with args: {sys.argv}")
    
    if len(sys.argv) < 5:
        print("Usage: python eval.py <model_path> <model_type> <data_path> <output_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].lower().strip()
    data_path  = sys.argv[3]
    output_path = sys.argv[4]

    print(f"DEBUG: Loading data from {data_path} with vocab for '{model_type}'...")
    
    # Pass model_type to load_data so it builds the correct vocab!
    input_data, output_data, vocab = load_data(data_path, model_type)
    
    print(f"DEBUG: Data loaded. input_len={len(input_data)}, vocab_size={len(vocab)}")

    if model_type == 'rnn':
        eval_rnn(model_path, input_data, output_data, output_path, vocab)
    elif model_type == 'transformer':
        eval_transformer(model_path, input_data, output_data, output_path, vocab)
    else:
        print(f"ERROR: Unknown model type '{model_type}'. Expected 'rnn' or 'transformer'.")