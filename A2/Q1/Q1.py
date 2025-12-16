import numpy as np
import pandas as pd
import re
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
np.random.seed(2)
nltk.download('stopwords')

class NaiveBayes:
    def __init__(self):
        self.alpha=1.0

    def calculate_class_prob(self):
        self.class_probs=np.zeros(self.num_classes)
        for i in self.df['Class Index']: self.class_probs[i]+=1
        self.class_probs/=self.num_training_samples
        # print(f"class probabilities = {self.class_probs}")

    def plot_wordcloud_per_class(self,path='plots'):
        os.makedirs(path,exist_ok=True)
        for i in range(self.num_classes):
            freqs={}
            for j in range(self.vocab_size):
                if self.word_counts[i][j]>0:
                    freqs[self.vocab[j]]=self.word_counts[i][j]
            wc=WordCloud(width=800,height=600,background_color='white',colormap='viridis').generate_from_frequencies(freqs)
            plt.figure(figsize=(10,8))
            plt.imshow(wc,interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.title(f'Class {i}')
            plt.savefig(os.path.join(path,f'class_{i}.png'))
            plt.close()

    def build_vocab(self,df,text_col):
        # build vocabulary
        all_words=[]
        for tokens in df[text_col]: 
            if tokens is not None:
                all_words.extend(tokens)
        vocab=list(set(all_words))
        vocab_size=len(vocab)
        vocab_dict={}
        print(f"vocab size: {vocab_size}")
        for i,word in enumerate(vocab): vocab_dict[word]=i
        return vocab,vocab_dict,vocab_size
    
    def cal_num_classes(self,class_col):
        classes={}
        for i in self.df[class_col]:
            if i not in classes:classes[i]=1
        return len(classes) 
    
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 0-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        self.df=df
        self.alpha=smoothening
        self.num_training_samples=self.df.shape[0]
        self.text_col=text_col
        self.num_classes=self.cal_num_classes(class_col)
        self.calculate_class_prob()
        self.vocab,self.vocab_dict,self.vocab_size=self.build_vocab(df,text_col)
        # init word count per class
        self.word_counts=np.zeros((self.num_classes,self.vocab_size))
        self.class_word_totals=np.zeros(self.num_classes)

        for _,row in self.df.iterrows():
            class_idx=row[class_col]
            for word in row[self.text_col]:
                if word in self.vocab_dict:
                    word_idx=self.vocab_dict[word]
                    self.word_counts[class_idx][word_idx]+=1
                    self.class_word_totals[class_idx]+=1
        # calc conditional prob P(word|class)
        self.word_likelihoods=((self.word_counts+self.alpha)/(self.class_word_totals[:,None]+self.alpha*self.vocab_size))

    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        predictions=[]
        self.text_col=text_col
        for tokens in df[self.text_col]:
            class_scores=np.log(self.class_probs+1e-9)

            for word in tokens:
                if word in self.vocab_dict:
                    word_idx=self.vocab_dict[word]
                    class_scores+=np.log(self.word_likelihoods[:,word_idx])
            predictions.append(np.argmax(class_scores))
        df[predicted_col]=predictions
        return df

class NaiveBayes2:
    def __init__(self):
        self.alpha=1.0

    def calculate_class_prob(self):
        self.class_probs=np.zeros(self.num_classes)
        for i in self.df['Class Index']: self.class_probs[i]+=1
        self.class_probs/=self.num_training_samples
        # print(f"class probabilities = {self.class_probs}")

    def plot_wordcloud_per_class(self,path='plots'):
        os.makedirs(path,exist_ok=True)
        for i in range(self.num_classes):
            freqs={}
            for j in range(self.vocab_size):
                if self.word_counts[i][j]>0:
                    freqs[self.vocab[j]]=self.word_counts[i][j]
            wc=WordCloud(width=800,height=600,background_color='white',colormap='viridis').generate_from_frequencies(freqs)
            plt.figure(figsize=(10,8))
            plt.imshow(wc,interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.title(f'Class {i}')
            plt.savefig(os.path.join(path,f'class_{i}.png'))
            plt.close()

    def build_vocab(self,df,text_col):
        # build vocabulary
        all_words=[]
        for tokens in df[text_col]: 
            if tokens is not None:
                all_words.extend(tokens)
        vocab=list(set(all_words))
        vocab_size=len(vocab)
        vocab_dict={}
        print(f"vocab size: {vocab_size}")
        for i,word in enumerate(vocab): vocab_dict[word]=i
        return vocab,vocab_dict,vocab_size
    
    def cal_num_classes(self,class_col):
        classes={}
        for i in self.df[class_col]:
            if i not in classes:classes[i]=1
        return len(classes) 
        
    def fit(self,df,smoothening, title_col="Title Tokens", content_col="Content Tokens", class_col="Class Index"):
        self.df=df
        self.alpha=smoothening
        self.num_training_samples=df.shape[0]
        self.num_classes=self.cal_num_classes(class_col)

        self.calculate_class_prob()

        self.vocab_title,self.vocab_dict_title,self.vocab_size_title=self.build_vocab(df,title_col)
        self.vocab_content,self.vocab_dict_content,self.vocab_size_content=self.build_vocab(df,content_col)

        self.word_counts_title=np.zeros((self.num_classes,self.vocab_size_title))
        self.class_word_totals_title=np.zeros(self.num_classes)

        self.word_counts_content=np.zeros((self.num_classes,self.vocab_size_content))
        self.class_word_totals_content=np.zeros(self.num_classes)

        for _,row in df.iterrows():
            y=row[class_col]

            for w in row[title_col]:
                if w in self.vocab_dict_title:
                    idx=self.vocab_dict_title[w]
                    self.word_counts_title[y,idx]+=1
                    self.class_word_totals_title[y]+=1

            for w in row[content_col]:
                if w in self.vocab_dict_content:
                    idx=self.vocab_dict_content[w]
                    self.word_counts_content[y,idx]+=1
                    self.class_word_totals_content[y]+=1

        self.word_likelihoods_title=(
            (self.word_counts_title+self.alpha)/(self.class_word_totals_title[:,None]+self.alpha*self.vocab_size_title)
        )
        self.word_likelihoods_content=(
            (self.word_counts_content+self.alpha)/(self.class_word_totals_content[:,None]+ self.alpha*self.vocab_size_content)
        )
    def predict(self,df,title_col="Title Tokens", content_col="Content Tokens", predicted_col="Predicted",lambda_title=1.0,lambda_content=1.0):
        predictions=[]
        for _,row in df.iterrows():
            class_scores=np.log(self.class_probs+1e-9)

            for w in row[title_col]:
                if w in self.vocab_dict_title:
                    idx=self.vocab_dict_title[w]
                    class_scores+=lambda_title*np.log(self.word_likelihoods_title[:,idx])

            for w in row[content_col]:
                if w in self.vocab_dict_content:
                    idx=self.vocab_dict_content[w]
                    class_scores+= lambda_content* np.log(self.word_likelihoods_content[:,idx])
            
            predictions.append(np.argmax(class_scores))
        df[predicted_col]=predictions
        return df

class NaiveBayes3:
    def __init__(self):
        self.alpha=1.0

    def calculate_class_prob(self):
        self.class_probs=np.zeros(self.num_classes)
        for i in self.df['Class Index']: self.class_probs[i]+=1
        self.class_probs/=self.num_training_samples
        # print(f"class probabilities = {self.class_probs}")

    def plot_wordcloud_per_class(self,path='plots'):
        os.makedirs(path,exist_ok=True)
        for i in range(self.num_classes):
            freqs={}
            for j in range(self.vocab_size):
                if self.word_counts[i][j]>0:
                    freqs[self.vocab[j]]=self.word_counts[i][j]
            wc=WordCloud(width=800,height=600,background_color='white',colormap='viridis').generate_from_frequencies(freqs)
            plt.figure(figsize=(10,8))
            plt.imshow(wc,interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.title(f'Class {i}')
            plt.savefig(os.path.join(path,f'class_{i}.png'))
            plt.close()

    def build_vocab(self,df,text_col):
        # build vocabulary
        all_words=[]
        for tokens in df[text_col]: 
            if tokens is not None:
                all_words.extend(tokens)
        vocab=list(set(all_words))
        vocab_size=len(vocab)
        vocab_dict={}
        print(f"vocab size: {vocab_size}")
        for i,word in enumerate(vocab): vocab_dict[word]=i
        return vocab,vocab_dict,vocab_size
    
    def cal_num_classes(self,class_col):
        classes={}
        for i in self.df[class_col]:
            if i not in classes:classes[i]=1
        return len(classes) 
        
    def get_class_freq(self,df):
        self.class_freq={}
        class_freq_table={}
        for i in range(df.shape[0]):
            cls=df["Class Index"][i]
            freq=df["Content Avg Length"][i]
            if cls not in class_freq_table: class_freq_table[cls]=[]
            class_freq_table[cls].append(freq)
        for i in range(self.num_classes):
            freq_values=class_freq_table[i]
            v=np.var(freq_values)+1e-4
            m=np.mean(freq_values)
            self.class_freq[i]={"mean":m, "var":v}

    def get_gll(self,val,mean,var):
        return -0.5*np.log(2*np.pi*var)-0.5*((val-mean)**2/var)
            
    def fit(self,df,smoothening, title_col="Title Tokens", content_col="Content Tokens", class_col="Class Index",avg_freq_col="Content Avg Length"):
        self.df=df
        self.alpha=smoothening
        self.num_training_samples=df.shape[0]
        self.num_classes=self.cal_num_classes(class_col)

        self.calculate_class_prob()
        self.get_class_freq(df)
        self.vocab_title,self.vocab_dict_title,self.vocab_size_title=self.build_vocab(df,title_col)
        self.vocab_content,self.vocab_dict_content,self.vocab_size_content=self.build_vocab(df,content_col)

        self.word_counts_title=np.zeros((self.num_classes,self.vocab_size_title))
        self.class_word_totals_title=np.zeros(self.num_classes)

        self.word_counts_content=np.zeros((self.num_classes,self.vocab_size_content))
        self.class_word_totals_content=np.zeros(self.num_classes)

        for _,row in df.iterrows():
            y=row[class_col]

            for w in row[title_col]:
                if w in self.vocab_dict_title:
                    idx=self.vocab_dict_title[w]
                    self.word_counts_title[y,idx]+=1
                    self.class_word_totals_title[y]+=1

            for w in row[content_col]:
                if w in self.vocab_dict_content:
                    idx=self.vocab_dict_content[w]
                    self.word_counts_content[y,idx]+=1
                    self.class_word_totals_content[y]+=1

        self.word_likelihoods_title=(
            (self.word_counts_title+self.alpha)/(self.class_word_totals_title[:,None]+self.alpha*self.vocab_size_title)
        )
        self.word_likelihoods_content=(
            (self.word_counts_content+self.alpha)/(self.class_word_totals_content[:,None]+ self.alpha*self.vocab_size_content)
        )

    def predict(self,df,title_col="Title Tokens", content_col="Content Tokens", predicted_col="Predicted",avg_freq_col="Content Avg Length",lambda_title=1.0,lambda_content=1.0):
        predictions=[]
        for _,row in df.iterrows():
            class_scores=np.log(self.class_probs+1e-9)
            freq_val=row[avg_freq_col]

            for w in row[title_col]:
                if w in self.vocab_dict_title:
                    idx=self.vocab_dict_title[w]
                    class_scores+=lambda_title*np.log(self.word_likelihoods_title[:,idx])

            for w in row[content_col]:
                if w in self.vocab_dict_content:
                    idx=self.vocab_dict_content[w]
                    class_scores+= lambda_content* np.log(self.word_likelihoods_content[:,idx])

            gll = np.array([self.get_gll(freq_val, self.class_freq[i]["mean"], self.class_freq[i]["var"])
                for i in range(self.num_classes)])
            
            class_scores+=gll
            predictions.append(np.argmax(class_scores))
        df[predicted_col]=predictions
        return df

def print_star(n=100):
    print('*'*n)

def get_tokens_from_string(text,is_lower=False,stem_stop=False,is_bigram=False):
    if is_lower: tokens=re.findall(r'\b\w+\b',str(text).lower())
    else: tokens=re.findall(r'\b\w+\b',str(text))
    if stem_stop:
        stemmer=PorterStemmer()
        stop_words=set(stopwords.words('english'))
        new_tokens=[stemmer.stem(word) for word in tokens if word not in stop_words]
        if not is_bigram: return new_tokens
    
    if is_bigram:
        if stem_stop:
            new_bigrams_tokens=[]
            for i in range(len(new_tokens)-1):
                bigram=new_tokens[i]+" "+new_tokens[i+1]
                new_bigrams_tokens.append(bigram)
            new_tokens.extend(new_bigrams_tokens)
            return new_tokens
        else:
            new_bigrams_tokens=[]
            for i in range(len(tokens)-1):
                bigram=tokens[i]+" "+tokens[i+1]
                new_bigrams_tokens.append(bigram)
            tokens.extend(new_bigrams_tokens)
            return tokens
        
    return tokens

def load_data(path,text_col,stem_stop=False,is_lower=False,is_bigram=False):
    df = pd.read_csv(path)
    num_training_samples=df.shape[0]
    print(f'number of examples: {num_training_samples}')
    df_new=pd.DataFrame()
    df_new["Class Index"]=df['label']
    tokens_col=[]
    for text in df[text_col]:tokens_col.append(get_tokens_from_string(text,is_lower,stem_stop,is_bigram))
    df_new["Tokenized Description"]=tokens_col
    # print(df_new.head())
    return df_new

def Q1(text_col):
    print_star()
    print("Q1. (a) implement nb using only content text. report training and test accuracy")
    print(f'Training of {text_col} column')
    start_time = time.time()
    train_df=load_data(train_path,text_col=text_col)
    test_df=load_data(test_path,text_col=text_col)
    nb=NaiveBayes()
    nb.fit(train_df,smoothening)

    pred_train_df=nb.predict(train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"]==pred_train_df["Class Index"])
    print(f"Training Accuracy: {accuracy_train:.4f}")

    pred_test_df=nb.predict(test_df)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy: {accuracy_test:.4f}")
    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')

    # Q1.b) construct word cloud representing the most frequent words for each class
    if text_col=="content": nb.plot_wordcloud_per_class(path='plots/Q1')
    if text_col=="title": nb.plot_wordcloud_per_class(path='plots/Q5.1')

    print_star()
    return pred_test_df

def Q2(text_col,stem_stop=True):
    print_star()
    print(" Q2. a) Perform stemming and remove the stop-words in the training as well as the validation data.")
    print(f'Training of {text_col} column')
    print("Perform stemming and remove the stop-words.")
    start_time=time.time()
    # i am getting (it the and) in wordclouds because is lower is false
    train_df=load_data(train_path,text_col=text_col,stem_stop=stem_stop)
    test_df=load_data(test_path,text_col=text_col,stem_stop=stem_stop)
    nbc=NaiveBayes()
    nbc.fit(train_df,smoothening)

    pred_train_df=nbc.predict(train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"]==pred_train_df["Class Index"])
    print(f"Training Accuracy of transformed set: {accuracy_train:.4f}")

    pred_test_df=nbc.predict(test_df)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy of transformed set: {accuracy_test:.4f}")
    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')

    # Q1.b) construct word cloud representing the most frequent words for each class
    if text_col=="content": nbc.plot_wordcloud_per_class(path='plots/Q1')
    if text_col=="title": nbc.plot_wordcloud_per_class(path='plots/Q5.2')

    print_star()
    return pred_test_df

def Q3(text_col,stem_stop=True):
    print_star()
    print("Train a model that utilizes both unigrams (individual words) and bigrams as features, ensuring that preprocessing from part (2) is applied beforehand.")
    print(f'Training of {text_col} column')

    print("Building both unigrams and bigrams as features after stemming and removing stop-words.")
    start_time=time.time()
    train_df=load_data(train_path,text_col=text_col,stem_stop=stem_stop,is_bigram=True)
    test_df=load_data(test_path,text_col=text_col,stem_stop=stem_stop,is_bigram=True)
    nbc=NaiveBayes()
    nbc.fit(train_df,smoothening)

    pred_train_df=nbc.predict(train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"]==pred_train_df["Class Index"])
    print(f"Training Accuracy of transformed set: {accuracy_train:.4f}")

    pred_test_df=nbc.predict(test_df)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy of transformed set: {accuracy_test:.4f}")
    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()
    return pred_test_df

def get_only_bigram_tokens(text):
    tokens=re.findall(r'\b\w+\b',str(text))
    new_bigrams_tokens=[]
    for i in range(len(tokens)-1):
        bigram=tokens[i]+" "+tokens[i+1]
        new_bigrams_tokens.append(bigram)
    return new_bigrams_tokens
def load_data_for_only_bigram(path,text_col):
    df = pd.read_csv(path)
    num_training_samples=df.shape[0]
    print(f'number of examples: {num_training_samples}')
    df_new=pd.DataFrame()
    df_new["Class Index"]=df['label']
    tokens_col=[]
    for text in df[text_col]:tokens_col.append(get_only_bigram_tokens(text))
    df_new["Tokenized Description"]=tokens_col
    # print(df_new.head())
    return df_new


def only_bigram(text_col):
    print_star()
    print("only bigram.")
    print(f'Training of {text_col} column')
    train_df=load_data_for_only_bigram(train_path,text_col=text_col)
    test_df=load_data_for_only_bigram(test_path,text_col=text_col)
    nbc=NaiveBayes()
    nbc.fit(train_df,smoothening)
    pred_train_df=nbc.predict(train_df)
    accuracy_train=np.mean(pred_train_df["Predicted"]==pred_train_df["Class Index"])
    print(f"Training Accuracy of transformed set: {accuracy_train:.4f}")

    pred_test_df=nbc.predict(test_df)
    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy of transformed set: {accuracy_test:.4f}")
    print_star()
    return pred_test_df


def Q4(text_col):
    # classifying based on the content text. Performance metrics such as 
    # accuracy, precision, recall, F1-score, or any other evaluation criteria.
    print_star()
    print("Q4. Analyze the performance of different models to identify which one works best for ")
    start_time=time.time()
    print(f'Training of {text_col} column')

    only_big=only_bigram(text_col=text_col)
    raw=Q1(text_col=text_col)
    unigram=Q2(text_col=text_col,stem_stop=True)
    uni_bigram_stem=Q3(text_col=text_col)
    bigram_no_stem=Q3(text_col=text_col,stem_stop=False)

    raw_acc=accuracy_score(raw['Class Index'],raw['Predicted'])
    raw_pre,raw_recall,raw_f1,_=precision_recall_fscore_support(raw['Class Index'],raw['Predicted'])
    print(f"Raw Accuracy: {raw_acc:.4f}")
    print(f"Raw Precision: {raw_pre}")
    print(f"Raw Recall: {raw_recall}")
    print(f"Raw F1-score: {raw_f1}")

    raw_acc=accuracy_score(only_big['Class Index'],only_big['Predicted'])
    raw_pre,raw_recall,raw_f1,_=precision_recall_fscore_support(only_big['Class Index'],only_big['Predicted'])
    print(f"Raw Accuracy: {raw_acc:.4f}")
    print(f"Raw Precision: {raw_pre}")
    print(f"Raw Recall: {raw_recall}")
    print(f"Raw F1-score: {raw_f1}")
    

    unigram_acc=accuracy_score(unigram['Class Index'],unigram['Predicted'])
    unigram_pre,unigram_recall,unigram_f1,_=precision_recall_fscore_support(unigram['Class Index'],unigram['Predicted'])
    print(f"unigram Accuracy: {unigram_acc:.4f}")
    print(f"unigram Precision: {unigram_pre}")
    print(f"unigram Recall: {unigram_recall}")
    print(f"unigram F1-score: {unigram_f1}")

    bigram_acc=accuracy_score(uni_bigram_stem['Class Index'],uni_bigram_stem['Predicted'])
    bigram_pre,bigram_recall,bigram_f1,_=precision_recall_fscore_support(uni_bigram_stem['Class Index'],uni_bigram_stem['Predicted'])
    print(f"Bigram Accuracy: {bigram_acc:.4f}")
    print(f"Bigram Precision: {bigram_pre}")
    print(f"Bigram Recall: {bigram_recall}")
    print(f"Bigram F1-score: {bigram_f1}")

    bigram_acc=accuracy_score(bigram_no_stem['Class Index'],bigram_no_stem['Predicted'])
    bigram_pre,bigram_recall,bigram_f1,_=precision_recall_fscore_support(bigram_no_stem['Class Index'],bigram_no_stem['Predicted'])
    print(f"Bigram with no stemming Accuracy: {bigram_acc:.4f}")
    print(f"Bigram with no stemming Precision: {bigram_pre}")
    print(f"Bigram with no stemming Recall: {bigram_recall}")
    print(f"Bigram with no stemming F1-score: {bigram_f1}")

    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()

def Q5():
    # Evaluating the Best Model for Title Features
    print_star()
    text_col="title"
    start_time=time.time()
    print(f'Training of {text_col} column')    
    Q1(text_col=text_col)
    Q2(text_col=text_col)
    Q3(text_col=text_col)
    Q4(text_col=text_col)
    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()

def Q6_a():
    # Using both title and content as features concatenating them.
    print_star()
    start_time=time.time()
    print(f'Implementing NBC with both title and content features by concatenating them.')
    title_col,content_col="title","content"

    title_train_df=load_data(train_path,text_col=title_col,stem_stop=False,is_bigram=True)
    title_test_df=load_data(test_path,text_col=title_col,stem_stop=False,is_bigram=True)

    content_train_df=load_data(train_path,text_col=content_col,stem_stop=False,is_bigram=True)
    content_test_df=load_data(test_path,text_col=content_col,stem_stop=False,is_bigram=True)

    new_train_df=pd.DataFrame()
    new_test_df=pd.DataFrame()
    l,l2="Tokenized Description","Class Index"
    new_train_df[l]=title_train_df[l]+content_train_df[l]
    new_test_df[l]=title_test_df[l]+content_test_df[l]
    new_train_df[l2]=title_train_df[l2]
    new_test_df[l2]=title_test_df[l2]
    print(new_train_df.head())
    print(new_test_df.head())
    nb=NaiveBayes()
    nb.fit(new_train_df,smoothening)

    pred_train_df=nb.predict(new_train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"]==pred_train_df["Class Index"])
    print(f"Training Accuracy: {accuracy_train:.4f}")

    pred_test_df=nb.predict(new_test_df)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy: {accuracy_test:.4f}")

    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()

def Q6_b():
        # Using both title and content as features seperately
    print_star()
    start_time=time.time()
    print(f'Implementing NBC with both title and content features by seperately.')
    title_col,content_col="title","content"

    title_train_df=load_data(train_path,text_col=title_col,stem_stop=False,is_bigram=True)
    title_test_df=load_data(test_path,text_col=title_col,stem_stop=False,is_bigram=True)

    content_train_df=load_data(train_path,text_col=content_col,stem_stop=False,is_bigram=True)
    content_test_df=load_data(test_path,text_col=content_col,stem_stop=False,is_bigram=True)
    # print(title_train_df.head())
    # print(content_test_df.head())

    train_df=pd.DataFrame()
    test_df=pd.DataFrame()
    title_col,content_col,class_idx="Title Tokens","Content Tokens","Class Index"
    l="Tokenized Description"
    train_df[class_idx]=title_train_df[class_idx]
    test_df[class_idx]=title_test_df[class_idx]

    train_df[title_col]=title_train_df[l]
    test_df[title_col]=title_test_df[l]

    train_df[content_col]=content_train_df[l]
    test_df[content_col]=content_test_df[l]

    # print(train_df.head())
    # print(test_df.head())
    nb=NaiveBayes2()
    nb.fit(train_df,smoothening)

    pred_train_df=nb.predict(train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"] == pred_train_df["Class Index"])
    print(f"Training Accuracy: {accuracy_train:.4f}")

    pred_test_df=nb.predict(test_df)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy: {accuracy_test:.4f}")

    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()
    return pred_test_df["Predicted"], pred_test_df["Class Index"]

def Q7():
    print_star()
    start_time=time.time()
    print(f'Implementing random and positive classifier')
    test_df=load_data(test_path,text_col="title")
    num_samples=test_df.shape[0]
    pred=[]
    for i in range(num_samples):pred.append(np.random.randint(0,num_classes))
    accuracy_test = np.mean(pred == test_df["Class Index"])
    print(f"Random classifier Test accuracy: {accuracy_test:.4f}")

    # positive classifier
    class_freq=test_df["Class Index"].value_counts()
    max_class=class_freq.idxmax()
    pred2=[]
    for i in range(num_samples): pred2.append(max_class)
    accuracy_test = np.mean(pred2 == test_df["Class Index"])
    print(f"Positive classifier Test accuracy: {accuracy_test:.4f}")


    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()

def Q8():
    print_star()
    print('Plotting confusion matrix')
    pred,y=Q6_b()
    cm=confusion_matrix(y,pred,labels=np.arange(0,14))
    p=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.arange(0,14))
    plt.figure(figsize=(10,10))
    p.plot(cmap='Blues',values_format="d")
    plt.title("Confusion Matrix for Combined Naive Bayes Model")
    plt.show()
    # plt.close()
    print_star()

def Q9():
    # Feature Engineering and Model Enhancement
    print_star()
    start_time=time.time()

    print(f'Implementing NBC Average Document Length Feature  with both title and content features by seperately.')
    title_col,content_col="title","content"

    title_train_df=load_data(train_path,text_col=title_col,stem_stop=False,is_bigram=True)
    title_test_df=load_data(test_path,text_col=title_col,stem_stop=False,is_bigram=True)

    content_train_df=load_data(train_path,text_col=content_col,stem_stop=False,is_bigram=True)
    content_test_df=load_data(test_path,text_col=content_col,stem_stop=False,is_bigram=True)
    # print(title_train_df.head())
    # print(content_test_df.head())

    train_df=pd.DataFrame()
    test_df=pd.DataFrame()
    title_col,content_col,class_idx="Title Tokens","Content Tokens","Class Index"
    l="Tokenized Description"
    train_df[class_idx]=title_train_df[class_idx]
    test_df[class_idx]=title_test_df[class_idx]

    train_df[title_col]=title_train_df[l]
    test_df[title_col]=title_test_df[l]

    train_df[content_col]=content_train_df[l]
    test_df[content_col]=content_test_df[l]

    avg_len_train=[]
    for tokens in train_df[content_col]:
        l=0
        if len(tokens)>0:
            for token in tokens:l+=len(token)
            l/=len(tokens)
        avg_len_train.append(l)
    train_df["Content Avg Length"]=avg_len_train

    avg_len_test=[]
    for tokens in test_df[content_col]:
        l=0
        if len(tokens)>0:
            for token in tokens:l+=len(token)
            l/=len(tokens)
        avg_len_test.append(l)
    test_df["Content Avg Length"]=avg_len_test

    # print(train_df.head())
    # print(test_df.head())

    nb=NaiveBayes3()
    nb.fit(train_df,smoothening)
    pred_train_df=nb.predict(train_df)

    accuracy_train=np.mean(pred_train_df["Predicted"] == pred_train_df["Class Index"])
    print(f"Training Accuracy: {accuracy_train:.4f}")
    lt,lc=2.0,1.0
    pred_test_df=nb.predict(test_df,lambda_title=lt,lambda_content=lc)

    accuracy_test = np.mean(pred_test_df["Predicted"] == pred_test_df["Class Index"])
    print(f"Test accuracy for lambda title= {lt} and lambda content= {lc} : {accuracy_test:.4f}")

    acc=accuracy_score(pred_test_df["Class Index"],pred_test_df["Predicted"])
    pre,recall,f1,_=precision_recall_fscore_support(pred_test_df["Class Index"],pred_test_df["Predicted"])
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    end_time=time.time()
    print(f'total time = {(end_time-start_time):.4f} seconds')
    print_star()

if __name__ == '__main__':
    train_path='data/train.csv'
    test_path='data/test.csv'
    num_classes=14
    smoothening=1
    # Q1(text_col="content")
    # Q2(text_col="content")
    # Q3(text_col="content")
    Q4(text_col="content")
    # Q5()
    # Q6_a()
    # Q6_b()
    # Q7()
    # Q8()
    # Q9()



    
# for alpha-0.1 test acc=0.9727
# for alpha-0.01 test acc=0.9714
#for alpha-0.05 test acc=0.9724
#for alpha-0.5 test acc=0.9717
'''
for alpha-0.1
Training Accuracy: 0.9987
Test accuracy for lambda title= 0.5 and lambda content= 0.5 : 0.9726
Test accuracy for lambda title= 0.5 and lambda content= 1.0 : 0.9722
Test accuracy for lambda title= 0.5 and lambda content= 2.0 : 0.9713
Test accuracy for lambda title= 1.0 and lambda content= 0.5 : 0.9729
Test accuracy for lambda title= 1.0 and lambda content= 1.0 : 0.9727
Test accuracy for lambda title= 1.0 and lambda content= 2.0 : 0.9723
Test accuracy for lambda title= 2.0 and lambda content= 0.5 : 0.9725
Test accuracy for lambda title= 2.0 and lambda content= 1.0 : 0.9731
Test accuracy for lambda title= 2.0 and lambda content= 2.0 : 0.9728

'''