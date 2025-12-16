import numpy as np 
import os 
import sys
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,accuracy_score
from neural_network import data_loader,open_img_to_numpy,NeuralNetwork
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class Logger:
    def __init__(self,file_path):
        self.terminal=sys.stdout
        self.log=open(file_path,"w",encoding="utf-8")

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

original_stdout = sys.stdout

def logging(s='1',device='cpu'):
    global current_logger
    os.makedirs("logs",exist_ok=True)
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)

def print_star(n=100):
    print("*"*n)

def plot(f1_scores_train,f1_scores_test,hidden_sizes):
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_sizes, f1_scores_train, marker='o', linewidth=2,label='Train F1_scores')
    plt.plot(hidden_sizes, f1_scores_test, marker='x', linewidth=2,label='Test F1_scores')
    plt.title("Average F1 Score vs Number of Hidden Units")
    plt.xlabel("Number of Hidden Units")
    plt.ylabel("Average F1 Score (Macro)")
    plt.grid(True)
    os.makedirs("plots",exist_ok=True)
    plot_path=os.path.join("plots","F1_score_vs_depth.png")
    plt.savefig(plot_path)
    # plt.show()

def save_result_in_csv(file_path,predictions,header='prediction'):
    predictions=[k+1 for k in predictions]
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    df=pd.DataFrame({header:predictions})

    if not os.path.exists(file_path) or os.path.getsize(file_path)==0:
        df.to_csv(file_path,index=False)
    else:
        df.to_csv(file_path,mode='a',index=False,header=False)

def main():
    logging('neural_c')

    start_time=time.time()
    print_star()
    print('Loading Data')
    _,train_path,test_path,output_path=sys.argv

    x_train,y_train,class_names=data_loader(train_path)
    x_test,y_test,_=data_loader(test_path)
    print(f'Data shape')
    print(f'X_train : {x_train.shape}, y_train : {y_train.shape}')
    print(f'X_test : {x_test.shape}, y_test : {y_test.shape}')
    csv_path=os.path.join(output_path,'prediction_c.csv')

    print_star()
    avg_f1_scores_train=[]
    avg_f1_scores_test=[]
    depth=4
    all_depths=[[512],[512,256],[512,256,128],[512,256,128,64]]
    for i in all_depths:
        print_star()
        a=time.time()
        print(f'Training for hidden layers = {i}')
        kwags_hps={
            'input_dim':x_train.shape[1],
            'output_dim':len(class_names),
            'hidden_layers':i,
            'learning_rate':1e-2,
            'batch_size':32,
            'epochs':350,
            'seed':42,
            'delta':1e-4,
            "patience": 5
        }
        print(f'Hyperparameters : {kwags_hps}')
    
        print_star()
        print(f'Training Network')
        nn=NeuralNetwork(kwags_hps)
        nn.fit(x_train, y_train, x_test, y_test)

        print_star()
        print(f'Testing Network Performance...')
        print()
        print(f'Training Results...')
        pred_train=nn.predict(x_train)
        print(f'train_acc : {accuracy_score(y_train,pred_train):.4f}')
        print()
        print("Classification Report of train data:")
        print(classification_report(y_train,pred_train,zero_division=0))
        print()
        f1_train=f1_score(y_train, pred_train, average='macro',zero_division=0)
        print(f"Train : precision = {precision_score(y_train, pred_train, average='macro',zero_division=0):.4f} "
            f"recall = {recall_score(y_train, pred_train, average='macro',zero_division=0):.4f} "
            f"f1 = {f1_train:.4f}")

        print()
        print(f'Testing Results...')
        preds_test=nn.predict(x_test)
        print(f'test_acc : {accuracy_score(y_test,preds_test):.4f}')

        print("Classification Report of test data:")
        print(classification_report(y_test,preds_test,zero_division=0))
        print()
        f1_test=f1_score(y_test, preds_test, average='macro',zero_division=0)
        print(f"Test  : precision = {precision_score(y_test, preds_test, average='macro',zero_division=0):.4f} "
            f"recall = {recall_score(y_test, preds_test, average='macro',zero_division=0):.4f} "
            f"f1 = {f1_test:.4f}")
        
        avg_f1_scores_train.append(f1_train)
        avg_f1_scores_test.append(f1_test)

        print()
        print(f'Saving predictions in csv...')
        save_result_in_csv(csv_path,preds_test)
        print_star()
        b=time.time()
        print(f'Total execution time for hidden units {i} is {((b-a)/60.0):.4f} minutes')
    print()
    print(f'Plotting avg f1 scores...')
    plot(avg_f1_scores_train,avg_f1_scores_test,[1,2,3,4])
    print_star()
    end_time=time.time()
    print(f'total execution time = {((end_time-start_time)/60.0):.4f} minutes')
    print_star()
    sys.stdout = original_stdout; current_logger.close()

if __name__=="__main__":
    main()