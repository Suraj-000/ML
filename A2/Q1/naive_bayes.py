import numpy as np
from helper_fns import print_star
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os 

# nltk.download('stopwords')

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
