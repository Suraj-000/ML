import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import f1_score
import os
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import copy
import sys
import time
from helper_fns import Logger,set_device,print_star,plot_maze,plot_training_curves

os.makedirs("logs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)

device=set_device()
# device=torch.device("cpu")
original_stdout = sys.stdout

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def logging(s='log'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)

def get_pad_tensor(batch,pad_idx=0):
    input,output=zip(*batch)
    i_len=[len(s) for s in input]
    o_len=[len(s) for s in output]

    max_input_len=max(i_len)
    max_output_len=max(o_len)

    padded_input=[]
    padded_target=[]

    for s in input:padded_input.append(s+[pad_idx]*(max_input_len-len(s)))
    for s in output:padded_target.append(s+[pad_idx]*(max_output_len-len(s)))

    return(
        torch.tensor(padded_input,dtype=torch.long),
        torch.tensor(i_len,dtype=torch.long),
        torch.tensor(padded_target,dtype=torch.long),
        torch.tensor(o_len,dtype=torch.long)
    )

def data_loader(train_path,test_path,batch_size,pad_idx,ratio=0.9,get_val=False,testing=False):
    train_dataset=MazeDataloader(train_path,build_vocab=True,testing=testing)
    vocab=train_dataset.vocab
    test_dataset=MazeDataloader(test_path,vocab=vocab,build_vocab=False,testing=testing)
    if get_val:
        train_len=int(len(train_dataset)*ratio)
        val_len=len(train_dataset)-train_len
        train_dataset,val_dataset=random_split(train_dataset,[train_len,val_len])

        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: get_pad_tensor(b,pad_idx),
        )
        test_loader=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: get_pad_tensor(b,pad_idx),
        )
        print(f'Data shape Train: {len(train_dataset)}')
        print(f'Data shape Validation: {len(val_dataset)}')
        return train_loader,test_loader,vocab,train_dataset,test_dataset
    else:
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: get_pad_tensor(b,pad_idx),
        )
        test_loader=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: get_pad_tensor(b,pad_idx),
        )

    print(f'Data shape Train: {len(train_dataset)}')
    print(f'Data shape Validation: {len(test_dataset)}')
    return train_loader,test_loader,vocab,train_dataset,test_dataset

def plot_5_pred(train_path,test_path,hps):
    pad_idx=hps['pad_idx']
    sos_idx=hps['sos_idx']
    eos_idx=hps['eos_idx']
    dropout=hps['dropout']
    train_loader,test_loader,vocab,train_dataset,test_dataset=data_loader(train_path,test_path,hps["batch_size"],pad_idx,testing=True)
    vocab_len=len(vocab)
    figs=[]
    encoder=EncoderRNN(vocab_len,hps['embedding_dim'],hps['hidden_dim'],num_layers=hps['nof_rnn_layers'],pad_idx=pad_idx,dropout=dropout)
    decoder=DecoderRNN(vocab_len,hps['embedding_dim'],encoder_hidden_dim=hps['hidden_dim'],decoder_hidden_dim=hps['hidden_dim'],num_layers=hps['nof_rnn_layers'],pad_idx=pad_idx,attention_hidden=hps['attention'],dropout=dropout)
    encoder=encoder.to(device)
    decoder=decoder.to(device)
    encoder.load_state_dict(torch.load(os.path.join('models','best_encoder_rnn.pth'),map_location=device))
    decoder.load_state_dict(torch.load(os.path.join('models','best_decoder_rnn.pth'),map_location=device))
    encoder.eval()
    decoder.eval()
    sample_indices=random.sample(range(len(test_dataset)),5)
    for sample_num,idx in enumerate(sample_indices,1):
        input_seq,output_seq=test_dataset[idx]
        input_tensor=torch.tensor(input_seq,dtype=torch.long).unsqueeze(0).to(device)
        input_len=torch.tensor([len(input_seq)],dtype=torch.long).to(device)

        with torch.no_grad():
            encoder_outputs,encoder_hidden=encoder(input_tensor,input_len)
            encoder_mask=(input_tensor!=pad_idx).to(device)

            decoder_init_hidden=encoder_hidden
            if decoder_init_hidden.size(0)!=decoder.num_layers:
                decoder_init_hidden=decoder_init_hidden[:decoder.num_layers]

            previous_token=torch.full((1,),sos_idx,dtype=torch.long,device=device)

            preds=[]
            max_output_len=50
            for t in range(max_output_len):
                logits,decoder_init_hidden,attention=decoder.forward(previous_token,decoder_init_hidden,encoder_outputs,encoder_mask)
                pred_token=torch.argmax(logits,dim=-1)
                preds.append(pred_token.item())
                previous_token=pred_token
                if pred_token.item()==eos_idx:break
        inv_vocab = {v:k for k,v in vocab.items()}
        input_tokens=[inv_vocab[idx] for idx in input_seq if idx !=pad_idx]
        target_tokens=[inv_vocab[idx] for idx in output_seq if idx!=pad_idx]
        pred_tokens=[inv_vocab[idx] for idx in preds if idx!=pad_idx]

        print(f"\n=== Sample {sample_num} ===")
        print(f"True Path:      {' '.join(target_tokens)}")
        print(f"Predicted Path: {' '.join(pred_tokens)}")
        
        maze_text=" ".join(input_tokens)
        pred_text="<PATH_START> " + " ".join(pred_tokens) + " <PATH_END>"
        target_text="<TARGETPATH_START> " + " ".join(target_tokens) + " <TARGETPATH_END>"
        full_tokens=(maze_text+" "+target_text+" "+pred_text).split()
        fig=plot_maze(full_tokens)
        figs.append(fig)

    save_path=os.path.join('plots',f'maze_predictions.png')
    combined_fig,axs=plt.subplots(1,5,figsize=(20,5))
    for ax,fig in zip(axs,figs):
        fig.canvas.draw()
        img=np.array(fig.canvas.renderer.buffer_rgba())
        ax.imshow(img,aspect='equal')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Maze prediction saved to plots/maze_predictions.png')

def pad_to_max_len(batch_list, pad_value, max_len=None):
    if max_len is None:
        max_len = max(x.size(1) for x in batch_list)

    padded = []
    for x in batch_list:
        diff = max_len - x.size(1)
        if diff > 0:
            pad = torch.full((x.size(0), diff), pad_value, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        padded.append(x)

    return torch.cat(padded, dim=0)

def seq_acc_with_eos(preds,targets,pad_idx,eos_idx):
    def strip(seq):
        out=[]
        for t in seq:
            if t==eos_idx:break
            if t!=pad_idx:out.append(t)
        return out
    correct=0
    total=preds.size(0)
    for i in range(total):
        p=strip(preds[i].tolist())
        t=strip(targets[i].tolist())
        if p==t:correct+=1
    return correct/total  

class MazeDataloader(Dataset):
    def __init__(self,path,vocab=None,pad_token="<PAD>", sos_token="<SOS>",eos_token="<EOS>",build_vocab=False,testing=False):
        self.data=pd.read_csv(path)
        if testing:
            self.data=self.data.iloc[:100]
            self.input_sequence=self.data['input_sequence'].apply(eval).tolist()
            self.output_sequence=self.data['output_path'].apply(eval).tolist()
            self.maze_type=self.data['maze_type']
        else:
            self.input_sequence=self.data['input_sequence'].apply(eval).tolist()
            self.output_sequence=self.data['output_path'].apply(eval).tolist()
            self.maze_type=self.data['maze_type']

        if build_vocab:self.vocab=self.build_vocab(self.input_sequence,self.output_sequence,pad_token,sos_token,eos_token)
        else: self.vocab=vocab
        self.vocab_len=len(self.vocab)

        self.pad_idx=self.vocab[pad_token]
        self.sos_idx=self.vocab[sos_token]
        self.eos_idx=self.vocab[eos_token]
    def build_vocab(self,input,output,pad,sos,eos):
        vocab={pad:0,sos:1,eos:2}
        idx=3
        for seq in input+output:
            for token in seq:
                if token not in vocab:
                    vocab[token]=idx
                    idx+=1
        return vocab
        
    def __len__(self):
        return len(self.input_sequence)
    
    def encode_sequence(self,seq):
        return [self.vocab[token] for token in seq]
    
    def __getitem__(self, index):
        input=self.encode_sequence(self.input_sequence[index])
        output=self.encode_sequence(self.output_sequence[index])+[self.eos_idx]
        return input,output
    
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

class TrainerRNN:
    def __init__(self,encoder,decoder,hps):
        self.encoder=encoder.to(device)
        self.decoder=decoder.to(device)
        self.pad_idx=hps['pad_idx']
        self.sos_idx=hps['sos_idx']
        self.eos_idx=hps['eos_idx']
        self.teacher_forcing_ratio=hps['teacher_forcing_ratio']
        self.lr=hps['lr']
        self.batch_size=hps['batch_size']
        self.epochs=hps['epochs']
        self.embedding_dim=hps['embedding_dim']
        self.hidden_dim=hps['hidden_dim']
        self.nof_rnn_layers=hps['nof_rnn_layers']

        params=list(encoder.parameters())+list(decoder.parameters())
        self.optimizer=optim.Adam(params,lr=self.lr)
        self.criterion=nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def train_epoch(self,dataloader):
        self.encoder.train()
        self.decoder.train()
        total_loss=0.0
        total_tokens=0
        correct_tokens=0
        seq_correct=0
        total_seq=0
        total_preds = []
        total_targets = []

        total_examples=0
        for i,batch in enumerate(dataloader,1):
            input,inp_len,target,target_len=batch
            input=input.to(device)
            target=target.to(device)
            inp_len=inp_len.to(device)
            target_len=target_len.to(device)

            max_target_len=target.size(1)
            batch_actual=input.size(0)
            self.optimizer.zero_grad()
            
            encoder_outputs,encoder_hidden=self.encoder(input,inp_len)
            encoder_mask=(input!=self.pad_idx).to(device)

            decoder_init_hidden=encoder_hidden
            if decoder_init_hidden.size(0)!=self.decoder.num_layers:
                decoder_init_hidden=decoder_init_hidden[:self.decoder.num_layers]

            previous_tokens=torch.full((batch_actual,),self.sos_idx,dtype=torch.long,device=device)

            all_preds=[]
            all_logits=[]
            for t in range(max_target_len):
                logits,decoder_init_hidden,attention=self.decoder.forward(previous_tokens,decoder_init_hidden,encoder_outputs,encoder_mask)
                all_logits.append(logits.unsqueeze(1))
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.unsqueeze(1))

                use_teacher=random.random()<self.teacher_forcing_ratio
                if use_teacher:previous_tokens=target[:,t]
                else: previous_tokens=preds
            
            all_logits = torch.cat(all_logits, dim=1)
            all_logits_flat = all_logits.reshape(-1, self.decoder.vocab_size)
            target_flat = target.reshape(-1)
        
            loss=self.criterion(all_logits_flat,target_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 2.0)

            self.optimizer.step()

            total_loss+=loss.item()*batch_actual
            total_examples+=batch_actual
            preds_tensor=torch.cat(all_preds,dim=1)

            total_preds.append(preds_tensor.reshape(-1))
            total_targets.append(target.reshape(-1))

            mask=(target!=self.pad_idx)
            matches=(preds_tensor==target)&mask
            correct=matches.sum().item()
            ntokens=mask.sum().item()
            total_tokens+=ntokens
            correct_tokens+=correct
            
            batch_seq_acc = seq_acc_with_eos(preds_tensor, target, self.pad_idx, self.eos_idx)
            seq_correct += batch_seq_acc * batch_actual
            total_seq += batch_actual

            # batch_f1 = compute_token_f1(preds_tensor.cpu(), targets_tensor.cpu(), self.pad_idx)

            # if i % 1000 == 0 or i == len(dataloader):
            #     batch_token_acc = (correct / (ntokens + 1e-12)) if ntokens > 0 else 0.0
                # print(f'   --> batch {i}/{len(dataloader)} |'
                #     f' Batch Loss: {loss.item():.4f} | '
                #     f'Batch Token Acc: {batch_token_acc*100:.2f}% | '
                #     f'Batch Seq Acc: {batch_seq_acc*100:.2f}% | '
                #     f'Batch F1: {batch_f1*100:.2f}%'
                #     )
        avg_loss=total_loss/total_examples
        token_accuracy=correct_tokens/(total_tokens+1e-12)
        seq_accuracy=seq_correct/(total_seq+1e-12)
        preds = torch.cat(total_preds).cpu()
        targets = torch.cat(total_targets).cpu()
        f1 = compute_token_f1(preds, targets, self.pad_idx)

        return avg_loss,token_accuracy,seq_accuracy,f1
    
    @torch.no_grad()
    def evaluate(self,dataloader):
        self.encoder.eval()
        self.decoder.eval()
        total_loss=0.0
        total_tokens=0
        correct_tokens=0
        seq_correct=0
        total_examples=0
        total_seq=0
        epoch_preds=[]
        epoch_targets=[]
        total_preds = []
        total_targets = []
        for i,batch in enumerate(dataloader,1):
            input,inp_len,target,target_len=batch
            input=input.to(device)
            target=target.to(device)
            inp_len=inp_len.to(device)
            target_len=target_len.to(device)
            max_target_len=target.size(1)

            encoder_outputs,encoder_hidden=self.encoder(input,inp_len)
            encoder_mask=(input!=self.pad_idx).to(device)

            decoder_init_hidden=encoder_hidden
            if decoder_init_hidden.size(0)!=self.decoder.num_layers:
                decoder_init_hidden=decoder_init_hidden[:self.decoder.num_layers]

            batch_actual=input.size(0)
            previous_tokens=torch.full((batch_actual,),self.sos_idx,dtype=torch.long,device=device)

            loss=0.0
            all_pred=[]
            all_targets=[]
            for t in range(max_target_len):
                logits,decoder_init_hidden,attention=self.decoder.forward(previous_tokens,decoder_init_hidden,encoder_outputs,encoder_mask)
                target_t=target[:,t]
                loss_t=self.criterion(logits,target_t)
                loss+=loss_t

                preds=torch.argmax(logits,dim=-1)
                all_pred.append(preds.unsqueeze(1))
                all_targets.append(target_t.unsqueeze(1))
                previous_tokens=preds
            
            loss=loss/max_target_len
            total_loss+=loss.item()*batch_actual
            total_examples+=batch_actual
            preds_tensor=torch.cat(all_pred,dim=1)
            targets_tensor=torch.cat(all_targets,dim=1)

            total_preds.append(preds_tensor.reshape(-1))
            total_targets.append(targets_tensor.reshape(-1))

            mask=(targets_tensor!=self.pad_idx)
            matches=(preds_tensor==targets_tensor)&mask
            correct=matches.sum().item()
            ntokens=mask.sum().item()
            total_tokens+=ntokens
            correct_tokens+=correct

            batch_seq_acc = seq_acc_with_eos(preds_tensor, targets_tensor, self.pad_idx, self.eos_idx)
            seq_correct += batch_seq_acc * batch_actual
            total_seq += batch_actual

        preds = torch.cat(total_preds).cpu()
        targets = torch.cat(total_targets).cpu()
        f1 = compute_token_f1(preds, targets, self.pad_idx)

        avg_loss=total_loss/total_examples
        token_accuracy=correct_tokens/(total_tokens+1e-12)
        seq_accuracy=seq_correct/(total_seq+1e-12)
        return avg_loss,token_accuracy,seq_accuracy,f1

def compute_token_f1(preds,targets,pad_idx):
    preds=preds.flatten()
    targets=targets.flatten()
    mask=(targets!=pad_idx)
    preds=preds[mask]
    targets=targets[mask]
    if preds.numel()==0:return 0.0
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    targets = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    return f1_score(targets, preds, average="micro")

def main(ab_path):
    print_star()
    start_time=time.time()
    print('Training Maze Path Finding using RNN...')
    print_star()

    hps={
        'batch_size':32,
        'epochs':20,
        'lr':1e-4,
        'dropout':0,
        'embedding_dim':128,
        'hidden_dim':512,
        'nof_rnn_layers':2,
        'optimizer':'adam',
        'pad_idx':0,
        'sos_idx':1,
        'eos_idx':2,
        'teacher_forcing_ratio':0.5,
        "attention":256
    }
    print(f'Hyperparameters : {hps}')
    print_star()
    pad_idx=hps['pad_idx']
    batch_size=hps['batch_size']
    print(f'loading data... ')

    train_path='COL774-A4-Maze-Dataset/train_6x6_mazes.csv'
    test_path='COL774-A4-Maze-Dataset/test_6x6_mazes.csv'
    train_loader,test_loader,vocab,train_dataset,test_dataset=data_loader(train_path,test_path,batch_size,pad_idx,get_val=False)
    vocab_len=len(vocab)
    print_star()

    print(f'Loading model...')
    encoder=EncoderRNN(vocab_len,hps['embedding_dim'],hps['hidden_dim'],num_layers=hps['nof_rnn_layers'],pad_idx=pad_idx,dropout=hps['dropout'])
    decoder=DecoderRNN(vocab_len,hps['embedding_dim'],encoder_hidden_dim=hps['hidden_dim'],decoder_hidden_dim=hps['hidden_dim'],num_layers=hps['nof_rnn_layers'],pad_idx=pad_idx,attention_hidden=hps['attention'],dropout=hps['dropout'])
    trainer=TrainerRNN(encoder,decoder,hps)
    print_star()

    print(f'Starting training...')
    train_losses,test_losses=[],[]
    train_tok_accs,val_tok_accs=[],[]
    train_seq_accs,val_seq_accs=[],[]
    train_f1s,val_f1s=[],[]

    num_epochs=hps['epochs']
    best_val_seq=-1.0
    for episode in range(num_epochs):
        a=time.time()
        t_loss,t_tok_acc,t_seq_acc,t_f1=trainer.train_epoch(train_loader)
        v_loss,v_tok_acc,v_seq_acc,v_f1=trainer.evaluate(test_loader)

        train_losses.append(t_loss)
        test_losses.append(v_loss)
        train_tok_accs.append(t_tok_acc)
        val_tok_accs.append(v_tok_acc)
        train_seq_accs.append(t_seq_acc)
        val_seq_accs.append(v_seq_acc)
        train_f1s.append(t_f1)
        val_f1s.append(v_f1)

        if v_seq_acc>best_val_seq:
            best_val_seq=v_seq_acc
            torch.save(encoder.state_dict(),os.path.join('models','best_encoder_rnn.pth'))
            torch.save(decoder.state_dict(),os.path.join('models','best_decoder_rnn.pth'))
            print(f'Best model saved at epoch {episode+1} with Val Seq Acc: {best_val_seq*100:.2f}%')
        b=time.time()
        print(f'--> '
            f'Ep {episode+1}/{num_epochs} | '
            f'Train Loss: {t_loss:.4f} | '
            f'Test Loss: {v_loss:.4f} | '
            f'Train Tok Acc: {t_tok_acc*100:.2f}% | '
            f'Val Tok Acc: {v_tok_acc*100:.2f}% | '
            f'Train Seq Acc: {t_seq_acc*100:.2f}% | '
            f'Val Seq Acc: {v_seq_acc*100:.2f}% | '
            f'Train F1: {t_f1*100:.2f}% | '
            f'Val F1: {v_f1*100:.2f}% | '
            f'Time: {((b-a)/60.0):.2f} minutes'
            )
    print_star()

    print('Plotting training curves...')
    plot_training_curves(train_losses,test_losses,train_seq_accs,val_seq_accs,train_f1s,val_f1s,train_tok_accs,val_tok_accs,save_path='plots/rnn_training_curves.png')
    print_star()

    print('Visualizing predictions for any 5 random mazes from test set...')
    plot_5_pred(train_path,test_path,hps)
    print_star()

    encoder_sd = torch.load("models/best_encoder_rnn.pth", map_location="cpu")
    decoder_sd = torch.load("models/best_decoder_rnn.pth", map_location="cpu")

    rnn_s2q={
        'encoder':encoder_sd,
        'decoder':decoder_sd
    }
    torch.save(rnn_s2q,'models/rnn.pth')
    
    print('Training completed.')
    print(f'Total Execution time is: {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()

# if __name__=="__main__":
    # logging("RNN")
    # set_seed(42)
    # absolute_path=os.getcwd()
    # main(absolute_path)
    # sys.stdout = original_stdout; current_logger.close()

    # print('Visualizing predictions for any 5 random mazes from test set...')
    # train_path='COL774-A4-Maze-Dataset/train_6x6_mazes.csv'
    # test_path='COL774-A4-Maze-Dataset/test_6x6_mazes.csv'
    # hps={
    #     'batch_size':32,
    #     'epochs':20,
    #     'lr':1e-4,
    #     'dropout':0,
    #     'embedding_dim':128,
    #     'hidden_dim':512,
    #     'nof_rnn_layers':2,
    #     'optimizer':'adam',
    #     'pad_idx':0,
    #     'sos_idx':1,
    #     'eos_idx':2,
    #     'teacher_forcing_ratio':0.5,
    #     "attention":256
    # }
    # plot_5_pred(train_path,test_path,hps)
    # print_star()
