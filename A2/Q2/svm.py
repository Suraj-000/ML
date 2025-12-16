from cvxopt import matrix,solvers
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV,KFold
import sys
from collections import Counter
from itertools import combinations


class SupportVectorMachine:
    def __init__(self):
        pass
    
    def _com_kernel(self,X1,X2):
        if self.kernel=='linear':return X1@X2.T
        else:
            X1_sq=np.sum(X1**2,axis=1).reshape(-1,1)
            x2_sq=np.sum(X2**2,axis=1).reshape(1,-1)
            dist=X1_sq+x2_sq-2*(X1@X2.T)
            return np.exp(-self.gamma*dist)
        

        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        num_samples,dim=X.shape

        self.gamma=gamma
        y=np.where(y==0,-1,1).astype(float)
        K=self._com_kernel(X,X)

        self.kernel=kernel
        P=matrix(np.outer(y,y)*K)
        q=matrix(-np.ones(num_samples))
        G=matrix(np.vstack((np.diag(-np.ones(num_samples)),np.eye(num_samples))))
        h=matrix(np.hstack((np.zeros(num_samples),C*np.ones(num_samples))))
        A=matrix(y,(1,num_samples))
        b=matrix(0.0)

        solvers.options['show_progress'] = False
        # solvers.options['abstol']=1e-3
        # solvers.options['reltol']=1e-3
        # solvers.options['feastol']=1e-3

        sol=solvers.qp(P,q,G,h,A,b)
        alphas=np.ravel(sol['x'])

        tol=1e-5

        sv_mask=alphas>tol
        self.sv=X[sv_mask]

        self.sv_alphas=alphas[sv_mask]
        self.sv_labels=y[sv_mask]
        self.num_sv=len(self.sv)

        margin_sv=np.where((alphas>tol)&(alphas<C-tol))[0]
        if len(margin_sv)>0:self.b=y[margin_sv[0]]-np.sum(self.sv_alphas*self.sv_labels*K[sv_mask][:,margin_sv[0]])
        else:
            K_sv=self._com_kernel(self.sv,self.sv)
            decision=(self.sv_alphas*self.sv_labels)@K_sv.T

            self.b=np.mean(self.sv_labels-decision)

        if kernel=='linear':self.w=np.sum((self.sv_alphas*self.sv_labels)[:,None]*self.sv,axis=0)
        else:  self.w=None 

    def predict(self, X):
        if self.kernel=='linear': df=X@self.w+self.b
        else:
            K=self._com_kernel(X,self.sv)
            df=(K@(self.sv_alphas*self.sv_labels))+self.b
        # return np.where(np.sign(df)==-1,0,1)
        return np.where(np.sign(df)==-1,0,1)

def print_star(n=100):
    print('*'*n)

def plot(img,title):
    os.makedirs('plots', exist_ok=True)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    # plt.show()
    path=os.path.join('plots',title)
    plt.savefig(path)

def plot_flat(img,title):
    img_reshaped=img.reshape(32,32,3)
    plot(img_reshaped,title)

def preprocess_image(img_path):
    img=Image.open(img_path)
    img=img.resize((32,32))
    img=np.array(img,dtype=np.float32)
    img/=255.0
    img=img.flatten()
    # plot_flat(img)
    # sys.exit()
    return img

def match_sv(sv1,sv2):
    match=0
    for i in sv1:
        if np.any(np.all(np.abs(sv2-i)<1e-6,axis=1)):match+=1
    return match

def load_images(data_directory, binary=False):
    X_train,y_train,X_test,y_test=[],[],[],[]
    all_classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    class_mapping={all_classes[i]:i for i in range(len(all_classes))}
    rollno=86
    a,b=rollno%10,(rollno+1)%10
    if binary:binary_map={all_classes[a]: 0,all_classes[b]: 1}  
    else:binary_map=None
    for phase in ['train','test']:
        phase_folder=os.path.join(data_directory,phase)

        for class_name, class_label in class_mapping.items():
            if binary and class_name not in binary_map: continue

            class_folder = os.path.join(phase_folder, class_name)

            for file_name in os.listdir(class_folder):
                img=preprocess_image(os.path.join(class_folder, file_name))
                if binary:label=binary_map[class_name]  
                else:label= class_label            
                
                if phase == 'train':
                    X_train.append(img)
                    y_train.append(label)
                else:
                    X_test.append(img)
                    y_test.append(label)
    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)

def Q1(kernel='linear'):
    print_star()
    print(f'Q1.1 SVM for binary classification using CVXOPT and linear kernel')    
    start_time=time.time()

    X_train, y_train, X_test, y_test = load_images(path, binary=True)
    
    svm=SupportVectorMachine()
    svm.fit(X_train,y_train,kernel=kernel)
    y_pred=svm.predict(X_test)
    acc=np.mean(y_pred==y_test)
    print(f"SVM binary cls cvxopt test accuracy = {acc:.4f}")
    print(f"nof of support vectors= {svm.num_sv}")
    sv=np.argsort(svm.sv_alphas)[-5:]
    sv_img=svm.sv[sv]
    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")
    print_star()

def Q2(path='data'):
    print_star()
    print(f'Q1.2 SVM for binary classification using CVXOPT and gaussian kernel')    
    start_time=time.time()

    X_train, y_train, X_test, y_test = load_images(path, binary=True)
    
    svm=SupportVectorMachine()
    svm.fit(X_train,y_train,kernel='gaussian')
    y_pred=svm.predict(X_test)
    acc=np.mean(y_pred==y_test)
    print(f"SVM binary cls cvxopt test accuracy = {acc:.4f}")
    print(f"nof of support vectors= {svm.num_sv}")
    sv=np.argsort(svm.sv_alphas)[-5:]
    sv_img=svm.sv[sv]

    svm_l=SupportVectorMachine()
    svm_l.fit(X_train,y_train,kernel="linear")
    sv_l=np.array(svm_l.sv)
    sv_g=np.array(svm.sv)

    match=0
    for i in sv_l:
        if np.any(np.all(np.abs(sv_g-i)<1e-6,axis=1)):match+=1
    print(f"Number of matching support vectors: {match}")

    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")
    print_star()

def Q3(path='data'):
    print_star()
    print(f'Q1.3 SVM for binary classification using LIBSVM with linear and gaussian kernel')    
    start_time=time.time()
    X_train, y_train, X_test, y_test = load_images(path, binary=True)
    
    svm_l=SVC(kernel="linear",C=C)
    svm_l.fit(X_train,y_train)
    y_pred_l=svm_l.predict(X_test)
    acc=accuracy_score(y_test,y_pred_l)
    print(f"SVM cls with LIBSVM with linear kernel test accuracy = {acc:.4f}")
    print(f"nof of support vectors= {svm_l.n_support_.sum()}")

    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")

    start_time=time.time()
    
    svm_g=SVC(kernel="rbf",C=C,gamma=gamma)
    svm_g.fit(X_train,y_train)
    y_pred_g=svm_g.predict(X_test)
    acc=accuracy_score(y_test,y_pred_g)
    print(f"SVM cls with LIBSVM with gaussian kernel test accuracy = {acc:.4f}")
    print(f"nof of support vectors= {svm_g.n_support_.sum()}")

    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")

    svm_cx_l=SupportVectorMachine()
    svm_cx_l.fit(X_train,y_train,kernel="linear")
    svm_cx_l_sv=np.array(svm_cx_l.sv)

    svm_cx_g=SupportVectorMachine()
    svm_cx_g.fit(X_train,y_train,kernel="gaussian")
    svm_cx_g_sv=np.array(svm_cx_g.sv)

    svm_lb_l_sv=np.array(svm_l.support_vectors_)
    svm_lb_g_sv=np.array(svm_g.support_vectors_)

    print('counting all matches between all four models')
    sv_models = {
        "CX Linear": svm_cx_l_sv,
        "CX Gaussian": svm_cx_g_sv,
        "LB Linear": svm_lb_l_sv,
        "LB Gaussian": svm_lb_g_sv
    }
    match_table = {}
    for name1, sv1 in sv_models.items():
        match_table[name1] = {}
        for name2, sv2 in sv_models.items():
            match_table[name1][name2] = match_sv(sv1, sv2)

    for k, v in match_table.items():
        print(k, v)
    print("(b) Compare weight (w), bias (b) obtained here with the first part for linear kernel.")
    w1,b1=svm_cx_l.w,svm_cx_l.b
    w2,b2= svm_l.coef_.flatten(),svm_l.intercept_[0]
    print(f"linear kernel CVXOPT w = {w1} b = {b1}")
    print(f"linear kernel LIBSVM sklear w = {w2} b = {b2}")
    wd=np.linalg.norm(w1-w2)
    print("Norm of weight difference:", wd)
    cos_sim=np.dot(w1,w2)/(np.linalg.norm(w1)*np.linalg.norm(w2))
    print("Cosine similarity between weights:", cos_sim)

    print_star()

def Q5(path='data'):
    print_star()
    print(f'Q1.5 One-vs-One Multi-Class SVM')    
    start_time=time.time()
    all_pairs=list(combinations(range(10),2))

    all_cls_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    X_train, y_train, X_test, y_test = load_images(path)

    one_v_one_m={}
    for i,(c1,c2) in enumerate(all_pairs,1):
        print(f'Training {i}/{len(all_pairs)} : classifier {c1} vs {c2}')
        a=time.time()
        mask=(y_train==c1)|(y_train==c2)
        X_c,y_c=X_train[mask],y_train[mask]
        y_encode=np.where(y_c==c1,0,1)

        model=SupportVectorMachine()

        model.fit(X_c,y_encode,kernel="gaussian")

        one_v_one_m[(c1,c2)]=model
        b=time.time()
        print(f'time taken for classifer {i} : {(b-a):.4f}')
    y_pred_list=[]
    for i,s in enumerate(X_test,1):
        vote_counts=[]
        a=time.time()
        for (c1,c2),d in one_v_one_m.items() :

            pred=d.predict( s.reshape(1, -1))[0]

            if pred==0:vote_counts.append(c1)  
            else: vote_counts.append(c2)
        most=Counter(vote_counts).most_common(1)[0][0]
        y_pred_list.append(most)
        b=time.time()
        print(f'time taken to predict sample {i} : {(b-a):.4f}')
    y_pred=np.array(y_pred_list)
    acc=np.mean(y_pred==y_test)

    print(f" OvO CVXOPT SVM test accuracy = {acc:.4f}")
    print(y_test.shape,y_pred.shape)
    # np.savez("svm_prediction.npz",y_test=y_test,y_pred=y_pred)
    
    cm=confusion_matrix(y_test,y_pred,labels=np.arange(1,11))
    p=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.arange(1,11))
    plt.figure(figsize=(10,10))
    p.plot(cmap='Blues',values_format="d")
    plt.title("Confusion Matrix for Combined Naive Bayes Model")
    plt.show()
    plt.savefig("cm_cvxopt_ovo.png")

    mis_idx = np.where(y_pred != y_test)[0]

    errors = Counter((y_test[i], y_pred[i]) for i in mis_idx)

    print("\nMost frequent misclassifications:")
    for (true_cls, pred_cls), count in errors.most_common(10):
        print(f"{all_cls_names[true_cls]} → {all_cls_names[pred_cls]} : {count} times")
    print("\nMost frequent misclassifications:")
    for (true_cls, pred_cls), count in errors.most_common(10):
        print(f"{all_cls_names[true_cls]} → {all_cls_names[pred_cls]} : {count} times")

    visualize_misclassified(X_test, y_test, y_pred, all_cls_names, n=10)

    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")
    print_star()
    
def visualize_misclassified(X_test, y_test, y_pred, class_names, n=10):
    mis_idx = np.where(y_pred != y_test)[0]
    if len(mis_idx) == 0:
        print("No misclassifications found!")
        return
    np.random.seed(42)
    sample_idx = np.random.choice(mis_idx, min(n, len(mis_idx)), replace=False)
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_idx, 1):
        img = X_test[idx].reshape(32, 32, 3)
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        plt.subplot(2, 5, i)
        plt.imshow(img)
        plt.title(f"T:{true_label}\nP:{pred_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/vis_miss_class.png')
    plt.show()

def Q6(path='data'):
    print_star()
    print(f'Q1.6 Multi-Class SVM using sklearn LIBSVM')    
    start_time=time.time()

    all_cls_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    cls_dict={j:i for i,j in enumerate(all_cls_names)}
    X_train, y_train, X_test, y_test = load_images(path)

    svm=SVC(kernel='rbf',C=C,gamma=gamma,decision_function_shape="ovo")
    svm.fit(X_train,y_train)
    y_pred=svm.predict(X_test)
    acc=np.mean(y_pred == y_test)

    print(f" OvO LIBSVM SVM test accuracy = {acc:.4f}")

    cm=confusion_matrix(y_test,y_pred,labels=np.arange(1,11))
    p=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.arange(0,14))
    plt.figure(figsize=(10,10))
    p.plot(cmap='Blues',values_format="d")
    plt.title("Confusion Matrix for Combined Naive Bayes Model")
    plt.show()
    plt.savefig("cm_libsvm_ovo.png")
    
    mis_idx = np.where(y_pred != y_test)[0]

    errors = Counter((y_test[i], y_pred[i]) for i in mis_idx)

    print("\nMost frequent misclassifications:")
    for (true_cls, pred_cls), count in errors.most_common(10):
        print(f"{all_cls_names[true_cls]} → {all_cls_names[pred_cls]} : {count} times")
    print("\nMost frequent misclassifications:")
    for (true_cls, pred_cls), count in errors.most_common(10):
        print(f"{all_cls_names[true_cls]} → {all_cls_names[pred_cls]} : {count} times")
    visualize_misclassified(X_test, y_test, y_pred, all_cls_names, n=10)


    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")
    print_star()

def Q8(path='data'):
    print_star()
    print("Q1.8 5 fold cross validation and hyperparameter tuning")
    start_time=time.time()


    c_grid=[10,5,1,1e-5,1e-3]
    cv_means,test_scores=[],[]

    X_train, y_train, X_test, y_test = load_images(path)
    for l,c in enumerate(c_grid):

        a=time.time()
        kf=KFold(n_splits=5,shuffle=True,random_state=504)
        fold_scores=[]
        svm=SVC(kernel='rbf',gamma=gamma,C=c,decision_function_shape='ovo')

        for idx,(t_idx,v_idx) in enumerate(kf.split(X_train,y_train),1):
            xtf,xvf=X_train[t_idx],X_train[v_idx]
            ytf,yvf=y_train[t_idx],y_train[v_idx]
            svm.fit(xtf,ytf)
            score=svm.score(xvf,yvf)
            fold_scores.append(score)
            print(f'Fold {idx} for C={c} score = {score}')

        m=np.mean(fold_scores)
        cv_means.append(m)

        f_svm=SVC(kernel='rbf',gamma=gamma,C=c,decision_function_shape='ovo')
        f_svm.fit(X_train,y_train)
        m=f_svm.score(X_test,y_test)
        test_scores.append(m)

        b=time.time()
        print(f'execution time for C={c} is {(b-a):.4f}')


    f=np.argmax(cv_means)
    print(f" final C = {c_grid[f]} with cross validation accuracy = {cv_means[f]:.4f} and test acc = {test_scores[f]:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(c_grid, test_scores, 's', label='Test Accuracy')

    plt.plot(c_grid, cv_means, 'o', label='CV Accuracy')
    plt.xscale('log')

    plt.ylabel('Accuracy')
    plt.xlabel('C (log scale)') 

    plt.title('SVM Accuracy vs C')
    plt.legend()

    plt.grid(True)
    plt.savefig("svm_c_tuning.png", bbox_inches='tight')
    # plt.show()
    print("Plot saved: plots/svm_c_tuning.png")

    end_time=time.time()
    print(f"total execution time : {(end_time-start_time):.4f}")
    print_star()

if __name__=="__main__":
    train_dir="data/train"
    test_dir="data/test"
    C=1.0
    path='data'
    gamma=0.001
    Q1()
    Q2()
    Q3()
    Q5()
    Q6()
    # Q7 is done in 5 and 6
    Q8(path='data')