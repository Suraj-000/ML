import numpy as np
import pandas as pd
import sys

class DecisionTree:
    def __init__(self,max_depth,criterion='entropy'):
        self.tree=None
        self.max_depth=max_depth
        self.criterion=criterion
    
    def fit(self,x,y,depth=0):
        vals,counts=np.unique(y,return_counts=True)
        if len(vals)==1: return vals[0]
        if depth>=self.max_depth:return vals[np.argmax(counts)]

        class_ig=[self.information_gain(x[attr],y) for attr in x.columns]
        d=np.argmax(class_ig)
        best_att=x.columns[d]
        majority_label=vals[np.argmax(counts)]

        tree_node={'att':best_att, "prediction":majority_label, 'children':{}}
        best_col_val=x[best_att]
        unique_vals=np.unique(best_col_val)

        if best_col_val.dtype=='object':
            for k in unique_vals:
                mask=(best_col_val==k)
                if np.sum(mask)==0:continue
                tree_node['children'][k]=self.fit(x[mask],y[mask],depth+1)
        else:
            median=np.median(best_col_val)
            tree_node['cont_split_val']=median
            lm=best_col_val<=median
            rm=best_col_val>median
            if np.sum(lm)>0:tree_node["children"]["<="]=self.fit(x[lm],y[lm],depth+1)
            if np.sum(rm)>0:tree_node["children"][">"]=self.fit(x[rm],y[rm],depth+1)

        if depth==0:self.tree=tree_node 
        return tree_node

    def entropy(self,data):
        y=len(data)
        if y==0:return 0
        val,class_counts=np.unique(data,return_counts=True)
        probs=class_counts/y
        return -np.sum(probs*np.log2(probs+1e-9))

    def gini_index(self,data):
        len_data=len(data)
        if len_data==0:return 0
        val,class_counts=np.unique(data,return_counts=True)
        probs=class_counts/len_data
        return 1-np.sum(probs**2)

    def information_gain(self,x_col,y):
        unique_vals=np.unique(x_col)
        t=len(y)
        if t==0:return 0
        if self.criterion=='gini_index':e_parent=self.gini_index(y)
        else:e_parent=self.entropy(y)

        cat_entropys_weighted=0
        if x_col.dtype=='object':subsets=[y[x_col==i] for i in unique_vals]
        else:
            median_val=np.median(x_col)
            l_m=x_col<=median_val
            r_m=~l_m
            subsets=[y[l_m], y[r_m]]
        for subset in subsets:
            if len(subset)>0:
                cat_weight=len(subset)/t
                if self.criterion=='gini_index':cat_entropys_weighted+=cat_weight*self.gini_index(subset)
                else:cat_entropys_weighted+=cat_weight*self.entropy(subset)
        return e_parent- cat_entropys_weighted

    def pred_single(self,x,node=None):
        if node is None:node=self.tree
        if not isinstance(node,dict) or 'children' not in node:
            return node if not isinstance(node,dict) else node.get('prediction',None)

        att_name=node['att']
        if att_name not in x.index:
            return node.get('prediction',None)

        val=x[att_name]
        if 'cont_split_val' in node:
            median=node['cont_split_val']
            branch='<=' if val <=median else '>'
        else:branch=val

        if 'children' not in node or branch not in node['children']:
            return node.get('prediction',None)
        return self.pred_single(x,node['children'][branch])

    def get_tree(self):
        return self.tree

    def predict(self,x_data):
        preds = []
        for _, row in x_data.iterrows():
            preds.append(self.pred_single(row))
        return np.array(preds)
    
    def init_pruning_params(self,node):
        if not isinstance(node,dict):return
        node['incorrect_preds']=0
        if 'children' in node:
            for child in node['children'].values():
                self.init_pruning_params(child)
    
    def update_pruning_params(self,node,x_row,true_label):
        if not isinstance(node, dict):return
        prediction=node.get('prediction')
        if prediction!=true_label:node['incorrect_preds']=node.get('incorrect_preds',0)+1

        if 'children' in node:
            att_name=node['att']
            if att_name in x_row.index:
                val=x_row[att_name]
                if 'cont_split_val' in node:branch = '<=' if val <= node['cont_split_val'] else '>'
                else:branch = val
                if branch in node['children']:self.update_pruning_params(node['children'][branch],x_row,true_label)
    
    def prune_with_counts(self,node,metrics,x_train,y_train,x_test,y_test,x_val,y_val):
        if not isinstance(node,dict):return 0
        if 'children' not in node:return node.get('incorrect_preds',0)
        child_incorrect=0
        for child in node['children'].values(): child_incorrect+=self.prune_with_counts(child,metrics,x_train,y_train,x_test,y_test,x_val,y_val)

        node_incorrect=node.get('incorrect_preds',0)
        if node_incorrect<=child_incorrect:
            node.pop('children',None)
            node.pop('cont_split_val',None)
            node['is_leaf']=True
            # metrics=calculate_acc(self,metrics,x_train,y_train,x_test,y_test,x_val,y_val)
            return node_incorrect
        return child_incorrect

def count_dtree_nodes(node):
    if type(node) is not dict:return 1
    count=1
    if "children" in node: 
        for child in node['children'].values():
            count+=count_dtree_nodes(child)
    return count

def dtree_accuracy(dtree, x, y):
    preds = dtree.predict(x)
    return np.mean(preds==y)

def calculate_acc(dtree,metrics,x_train,y_train,x_test,y_test,x_val,y_val):
    metrics['nodes'].append(count_dtree_nodes(dtree.tree))
    metrics['train_acc'].append(dtree_accuracy(dtree, x_train, y_train))
    metrics['val_acc'].append(dtree_accuracy(dtree, x_val, y_val))
    metrics['test_acc'].append(dtree_accuracy(dtree, x_test, y_test))
    return metrics