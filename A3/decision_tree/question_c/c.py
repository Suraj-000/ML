import sys,os
import time
import pandas as pd
import csv
import numpy as np
import pprint
import copy
from datetime import datetime
import matplotlib.pyplot as plt

os.makedirs("logs",exist_ok=True)
log_path=os.path.join('logs','output.txt')
class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")
        self.log.write("\n\n--- Run started at: " + str(datetime.now()) + " ---\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(log_path)
print("Logging started — all terminal output will also go to:", log_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from decision_tree import DecisionTree


def print_star(n=100):
    print("*"*n)

def save_result_in_csv(path,y_test_pred,header='result'):
    data={header:y_test_pred}
    df=pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path,index=False)
    print(f'Data saved to :{path}')

def accuracy(y_true,y_pred):
    return np.mean(y_true==y_pred)

def dtree_accuracy(dtree,x,y):
    y_pred=dtree.predict(x)
    return accuracy(y_pred,y)


def load_data(path):
    ''' 
    path: path to csv file
    '''
    df=pd.read_csv(path)
    if 'result' not in df.columns: 
        print(f'column named results not in csv file')
        sys.exit()
    y=df['result']
    x=df.drop('result',axis=1)
    categorical_att=[]
    for att in x.columns:
        b=np.unique(x[att])
        if x[att].dtype=='object': categorical_att.append(att)
    x_onehot=pd.get_dummies(x,columns=categorical_att,drop_first=False).astype(int)
    return x_onehot,y

def visualise_tree_dfs(node,depth=0):
    if not isinstance(node,dict):
        print("  "*depth+f"→ {node}")
        return
    print("  "*depth+f"[{node['att']}] (pred={node['prediction']})")

    for key, child in node['children'].items():
        print("  " * (depth+1) + f"({key})")
        visualise_tree_dfs(child, depth+2)

def replace_subtree(tree,target_node,new_val):
    if tree is target_node :return new_val
    if type(tree) is not dict or 'children' not in tree:return tree
    for key,child in list(tree['children'].items()):
        r=replace_subtree(child,target_node,new_val)
        if r is not child:tree['children'][key]=r
    return tree

def count_dtree_nodes(node):
    if type(node) is not dict:return 1
    count=1
    if "children" in node: 
        for child in node['children'].values():
            count+=count_dtree_nodes(child)
    return count

def calculate_all_accuracies(dtree,metrics,x_train,y_train,x_test,y_test,x_val,y_val):
    # if count_dtree_nodes(dtree.tree) not in metrics['nodes']
    metrics['nodes'].append(count_dtree_nodes(dtree.tree))
    metrics['train_acc'].append(dtree_accuracy(dtree, x_train, y_train))
    metrics['val_acc'].append(dtree_accuracy(dtree, x_val, y_val))
    metrics['test_acc'].append(dtree_accuracy(dtree, x_test, y_test))
    return metrics


def post_pruning_tree(dtree,x_train,y_train,x_test,y_test,x_val,y_val,ep=1e-3):
    metrics = {
        'nodes': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    metrics=calculate_all_accuracies(dtree,metrics,x_train,y_train,x_test,y_test,x_val,y_val)
    dtree.init_pruning_params(dtree.tree)
    for idx in range(len(x_val)):dtree.update_pruning_params(dtree.tree,x_val.iloc[idx],y_val.iloc[idx])

    dtree.prune_with_counts(dtree.tree,metrics,x_train,y_train,x_test,y_test,x_val,y_val)
    metrics=calculate_all_accuracies(dtree,metrics,x_train,y_train,x_test,y_test,x_val,y_val)
    return metrics

def plot_new(metrics,i):
    fig,ax=plt.subplots(figsize=(10,8))
    ax.plot(metrics['nodes'],metrics['train_acc'],'o-',label='Training',linewidth=2,markersize=4)
    ax.plot(metrics['nodes'],metrics['val_acc'],'s-',label='Validation',linewidth=2,markersize=4)
    ax.plot(metrics['nodes'],metrics['test_acc'],'^-',label='Test',linewidth=2,markersize=4)
    ax.set_xlabel('Number of Nodes',fontsize=12)
    ax.set_ylabel('Accuracy',fontsize=12)
    ax.set_title(f'Depth {i} : Accuracy vs Tree Size During Pruning',fontsize=14,fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True,alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plot_path=os.path.join("../plots",f"Q3_accuracies_vs_nof_nodes_{i}.png")
    plt.savefig(plot_path)
    # plt.show()
    plt.close()
    # print(f'image saved in plots folder')

def main():
    print_star()
    start_time=time.time()
    print(f"Q3. Decision tree construction with One hot encoding and Post Pruning.")
    max_depths=[15,25,35,45]
    _,train_path,test_path,val_path,output_path=sys.argv
    x_train,y_train=load_data(train_path)
    print(f'Number of training samples: {x_train.shape[0]}')
    x_test,y_test=load_data(test_path)
    print(f'Number of testing samples: {x_test.shape[0]}')
    x_val,y_val=load_data(val_path)
    print(f'Number of validation samples: {x_val.shape[0]}')
    testPredictions=[]
    for i,d in enumerate(max_depths):
        print_star(80)
        print(f'Implementing decison tree for max depth {d}')
        a=time.time()
        dtree=DecisionTree(max_depth=d)
        dtree.fit(x_train,y_train)
        
        print(f'before pruning best validation accuracy = {dtree_accuracy(dtree,x_val,y_val)}')
        metrics=post_pruning_tree(dtree,x_train,y_train,x_test,y_test,x_val,y_val)
        print(f'after pruning best validation accuracy = {dtree_accuracy(dtree,x_val,y_val)}')
        y_test_pred=dtree.predict(x_test)
        testPredictions.append(y_test_pred)
        init_nodes,final_nodes=metrics['nodes'][0],metrics['nodes'][-1]
        print(f'initial nodes = {init_nodes} final nodes = {final_nodes} nof nodes pruned = {init_nodes-final_nodes} ')
        plot_new(metrics,d)
        b=time.time()-a
        print(f'total time of execution for depth {d} : {b:.4f}s')
        print_star(80)
    save_result_in_csv(output_path,testPredictions[2])
    print(f'Pruned tree obtained after pruning the max depth 35 tree and is saved in output folder')

    end_time=time.time()
    print(f'total execution time = {(end_time-start_time):.4f}')
    print_star()

if __name__=='__main__':
    main()