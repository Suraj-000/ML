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

def plot(f1_scores_train,f1_scores_test,hidden_sizes,s='f1'):
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_sizes, f1_scores_train, marker='o', linewidth=2,label='Train F1_scores')
    plt.plot(hidden_sizes, f1_scores_test, marker='x', linewidth=2,label='Test F1_scores')
    plt.title("F1 Score vs epochs")
    plt.ylabel("epochs")
    plt.xlabel("F1 Score")
    plt.grid(True)
    os.makedirs("plots",exist_ok=True)
    plot_path=os.path.join("plots",f"F1_score_vs_epochs_{s}.png")
    plt.savefig(plot_path)
    # plt.show()

def save_result_in_csv(file_path,predictions,header='prediction'):
    predictions=[k+37 for k in predictions]
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    df=pd.DataFrame({header:predictions})

    if not os.path.exists(file_path) or os.path.getsize(file_path)==0:
        df.to_csv(file_path,index=False)
    else:
        df.to_csv(file_path,mode='a',index=False,header=False)

def get_model(model_path):
    start_time=time.time()
    print_star()
    print('Loading Data')
    _,train_path,test_path,output_path=sys.argv

    x_train,y_train,class_names=data_loader(train_path)
    x_test,y_test,_=data_loader(test_path)
    print(f'Data shape')
    print(f'X_train : {x_train.shape}, y_train : {y_train.shape}')
    print(f'X_test : {x_test.shape}, y_test : {y_test.shape}')
    csv_path=os.path.join(output_path,'prediction_f.csv')

    print_star()
    all_depths=[512,256,128,64]
    print_star()
    print(f'Training for hidden layers = {all_depths}')
    kwags_hps={
        'input_dim':x_train.shape[1],
        'output_dim':len(class_names),
        'hidden_layers':[512,256,128,64],
        'learning_rate':1e-2,
        'batch_size':32,
        'epochs':150,
        'seed':42,
        'delta':1e-4,
        "patience": 4,
        "activation":'relu'

    }
    print(f'Hyperparameters : {kwags_hps}')

    print_star()
    print(f'Training Network')
    nn=NeuralNetwork(kwags_hps)
    nn.fit(x_train, y_train, x_test, y_test)

    os.makedirs("models", exist_ok=True)
    nn.save_weights(model_path)
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
    print()
    print_star()
    end_time=time.time()
    print(f'total execution time = {((end_time-start_time)/60.0):.4f} minutes')
    print_star()

def testing_model(model_path):
    print_star()
    print('Loading Data')
    _,train_path,test_path,output_path=sys.argv

    x_train,y_train,class_names=data_loader(train_path)
    x_test,y_test,_=data_loader(test_path)
    print(f'Data shape')
    print(f'X_train : {x_train.shape}, y_train : {y_train.shape}')
    print(f'X_test : {x_test.shape}, y_test : {y_test.shape}')
    csv_path=os.path.join(output_path,'prediction_f.csv')

    print_star()
    all_depths=[512,256,128,64]
    print_star()
    print(f'Training for hidden layers = {all_depths}')

    kwags_hps={
            'input_dim':x_train.shape[1],
            'output_dim':len(class_names),
            'hidden_layers':[512,256,128,64],
            'learning_rate':1e-2,
            'batch_size':32,
            'epochs':5,
            'seed':42,
            'delta':1e-4,
            "patience": 4,
            "activation":'relu'

        }

    nn2 = NeuralNetwork(kwags_hps)

    nn2.load_weights(model_path)
    print('testing loaded model')
    print(nn2.get_acc(x_test,y_test))

def part1():
    start_time=time.time()
    print_star()
    print('Loading Data')
    _,train_path,test_path,output_path=sys.argv

    x_train,y_train,class_names=data_loader(train_path)
    x_test,y_test,_=data_loader(test_path)
    print(f'Data shape')
    print(f'X_train : {x_train.shape}, y_train : {y_train.shape}')
    print(f'X_test : {x_test.shape}, y_test : {y_test.shape}')
    csv_path=os.path.join(output_path,'prediction_f.csv')

    print_star()
    all_depths=[512,256,128,64]
    print_star()
    print(f'Training for hidden layers = {all_depths}')
    kwags_hps={
        'input_dim':x_train.shape[1],
        'output_dim':len(class_names),
        'hidden_layers':[512,256,128,64],
        'learning_rate':1e-2,
        'batch_size':32,
        'epochs':20,
        'seed':42,
        'delta':1e-4,
        "patience": 4,
        "activation":'relu'

    }
    print(f'Hyperparameters : {kwags_hps}')

    print_star()
    print(f'Training Network')
    nn=NeuralNetwork(kwags_hps)
    nn.fit(x_train, y_train, x_test, y_test)
    f1_scores_train,f1_scores_test=nn.get_f1_20()

    print()
    print(f'plotting f1 scores test and train..')
    plot(f1_scores_train,f1_scores_test,np.arange(1, 21))

    print_star()
    print(f'Testing Network Performance...')
    print()
    print(f'Training Results...')
    pred_train=nn.predict(x_train)
    print(f'train_acc : {accuracy_score(y_train,pred_train):.4f}')
    print()
    print(f'Testing Results...')
    preds_test=nn.predict(x_test)
    print(f'test_acc : {accuracy_score(y_test,preds_test):.4f}')
    print(f'precision : {precision_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print(f'recall : {recall_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print(f'F1 score : {f1_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print()
    print(f'Saving predictions in csv...')
    save_result_in_csv(csv_path,preds_test)
    print_star()
    end_time=time.time()
    print(f'total execution time = {((end_time-start_time)/60.0):.4f} minutes')
    print_star()

def main(model_path):
    start_time=time.time()
    print_star()
    print('Loading Data')
    _,train_path,test_path,output_path=sys.argv

    x_train,y_train,class_names=data_loader(train_path)
    x_test,y_test,_=data_loader(test_path)
    print(f'Data shape')
    print(f'X_train : {x_train.shape}, y_train : {y_train.shape}')
    print(f'X_test : {x_test.shape}, y_test : {y_test.shape}')
    csv_path=os.path.join(output_path,'prediction_f.csv')

    pre_hps={
        'input_dim':x_train.shape[1],
        'output_dim':len(class_names),
        'hidden_layers':[512,256,128,64],
        'learning_rate':1e-2,
        'batch_size':32,
        'epochs':20,
        'seed':42,
        'delta':1e-4,
        "patience": 3,
        "activation":'relu'

    }
    print(f'loading pretrained model')
    model=NeuralNetwork(pre_hps)
    model.load_for_transfer(model_path,len(class_names))
    print_star()
    print(f'Fine-tuning on digit dataset')
    model.fit(x_train,y_train,x_test,y_test)

    train_f1,test_f1=model.get_f1_20()
    plot(train_f1,test_f1,np.arange(1, 21),'transfer_f1')
    print_star()
    print(f'Testing Network Performance...')
    print()
    print(f'Training Results...')
    pred_train=model.predict(x_train)
    print(f'train_acc : {accuracy_score(y_train,pred_train):.4f}')
    print()
    print(f'Testing Results...')
    preds_test=model.predict(x_test)
    print(f'test_acc : {accuracy_score(y_test,preds_test):.4f}')
    print(f'precision : {precision_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print(f'recall : {recall_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print(f'F1 score : {f1_score(y_test,preds_test,average='macro',zero_division=0):.4f}')
    print()
    print(f'Saving predictions in csv...')
    save_result_in_csv(csv_path,preds_test)
    print_star()
    end_time=time.time()
    print(f'total execution time = {((end_time-start_time)/60.0):.4f} minutes')
    print_star()

if __name__=="__main__":
    model_path = os.path.join("models", "nn_weights_relu_f.npz")
    # get_model(model_path)
    # testing_model(model_path)
    # logging('neural_f')
    part1()
    main(model_path)
    # sys.stdout = original_stdout; current_logger.close()