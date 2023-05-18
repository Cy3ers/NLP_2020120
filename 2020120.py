#!/usr/bin/env python
# coding: utf-8

# # NLP_FINAL

# #### Importing Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

from wordcloud import WordCloud
from tqdm.auto import tqdm
import matplotlib.style as style
style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")


# #### Loading CSV

# In[2]:


df = pd.read_csv('tweets.csv', encoding='utf8')
df = df.head(500)


# In[3]:


df.head()


# In[4]:


df['user_followers'] = pd.to_numeric(df['user_followers'], errors='coerce')

# Just converts all str types in the column to int so that comparison operator can be used


# In[5]:


df = df.drop_duplicates(keep='first')

# Removes duplicate rows as per instructions


# #### Checking Dataset

# In[6]:


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


# ##### Percentages of missing values per column

# In[7]:


(df.isnull().sum() / len(df)) * 100


# In[8]:


df = df.drop(['user_location', 'hashtags', 'user_description'], axis=1)

# Dropping rows with missing values as per instructions


# #### Filling NaN Values
df = df.dropna() # Not needed yetdf.head(2)
# #### Checking some features

# In[9]:


print(len(df[df['user_followers'] < 1000]), 'users with less than 1000 followers')
print(len(df[df['user_followers'] > 1000]), 'users with more than 1000 followers')


# ### Cleaning functions

# In[10]:


def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   
    
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)

#Break sentences into individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens

#Break paragraphs into sentence tokens
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    

def tokenize(text):
    return re.findall(r'\w+', text)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)


# Calling all the cleaning functions
def clean_text(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text


# In[11]:


# This was so that TQDM does not give AttributeError

df["text"] = df["text"].astype(str)
df["text"] = [x.replace(':',' ') for x in df["text"]]


# ### Running text cleaner and storing in new column

# In[12]:


df['clean_text'] = pd.Series([clean_text(i) for i in tqdm(df['text'])])


# In[13]:


words = df["clean_text"].values


# In[14]:


ls = []

for i in words:
    ls.append(str(i))


# In[15]:


ls[:5]


# ### Wordcloud 

# In[16]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="lightblue", colormap='Set2', max_words=1000, max_font_size= 200,  width=1600, height=800)
wc.generate(" ".join(ls))
plt.title("Most discussed terms", fontsize=20)
plt.imshow(wc.recolor( colormap= 'Set2' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')


# Trying some Features

# In[17]:


most_pop = df.sort_values('user_followers', ascending =False)[['user_name', 'user_followers']].head(12)

most_pop['user_followers1'] = most_pop['user_followers']/1000


# In[18]:


most_pop


# In[19]:


plt.figure(figsize = (20,25))

sns.barplot(data = most_pop, y = 'user_name', x = 'user_followers1', color = 'c')
plt.xticks(fontsize=27, rotation=0)
plt.yticks(fontsize=30, rotation=0)
plt.xlabel('User followers in Thousands', fontsize = 21)
plt.ylabel('')
plt.title('Followers', fontsize = 30);


# ## N-Gram Analysis

# In[20]:


def textNgrams(text, size):
    ngrams_all = []
    for string in text:
        tokens = string.split()
        if len(tokens) <= size:
            continue
        else:
            output = list(ngrams(tokens, size))
        for ngram in output:
            ngrams_all.append(" ".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    df_Ngms = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    df_Ngms = df_Ngms.rename(columns={'index':'words', 0:'count'})
    df_Ngms = df_Ngms.sort_values(by='count', ascending=False)
    df_Ngms = df_Ngms.head(10)
    df_Ngms = df_Ngms.sort_values(by='count')
    
    return(df_Ngms)


# In[21]:


def plotNgrams(text):
    bigrams = textNgrams(text, 2)
    trigrams = textNgrams(text, 3)
    
    # Set plot figure size
    fig = plt.figure(figsize = (20, 7))
    plt.subplots_adjust(wspace=.5)

    ax2 = fig.add_subplot(132)
    ax2.barh(np.arange(len(bigrams['words'])), bigrams['count'], align='center', alpha=.5)
    ax2.set_title('Bigrams')
    plt.yticks(np.arange(len(bigrams['words'])), bigrams['words'])
    plt.xlabel('Count')

    ax3 = fig.add_subplot(133)
    ax3.barh(np.arange(len(trigrams['words'])), trigrams['count'], align='center', alpha=.5)
    ax3.set_title('Trigrams')
    plt.yticks(np.arange(len(trigrams['words'])), trigrams['words'])
    plt.xlabel('Count')

    plt.show()


# In[22]:


def textTrends(text):
    plotNgrams(text)


# In[23]:


textTrends(df["clean_text"])


# ### Sentiment Analysis

# In[24]:


from textblob import TextBlob


# In[25]:


df_Copy = df


# In[26]:


df_Copy['polarity'] = df.clean_text.apply(lambda x: TextBlob(x).polarity)
df_Copy['subjectivity'] = df.clean_text.apply(lambda x: TextBlob(x).subjectivity)


# In[27]:


df['sentiment'] = np.where(df_Copy.polarity >= 0.05, 'Positive', 
                                 np.where(df_Copy.polarity <= 0.05, 'Negative', 'Neutral'))


# In[28]:


df.head()


# ### Feature Engineering

# Sentence Length

# In[29]:


df['sent_length'] = df['text'].apply(
    lambda row: min(len(row.split(" ")), len(row)) if isinstance(row, str) else None
)


# In[30]:


df.head(1)


# Number of words in a text

# In[31]:


df['word_count'] = df['text'].str.split().str.len()


# In[32]:


df.head(1)


# Number of spaces

# In[33]:


df['space_count'] = df['text'].str.count(' ')


# In[34]:


df.head(1)


# Number of characters

# In[35]:


df['char_count'] = df['text'].str.len()


# In[36]:


df.head(1)


# # Vectorization

# ### 1. Count Vectorizer

# In[37]:


from sklearn.feature_extraction.text import CountVectorizer


# In[38]:


count_vectorizer = CountVectorizer()
count_vectorizer.fit(df['text'])
count_vectorizer.vocabulary_


# ### 2. TD/IDF

# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[40]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(df['text'])
tfidf_vectorizer.vocabulary_


# ### 3. Word2Vec

# In[41]:


import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# In[42]:


#for textblob dataframe and vader dataframe
tagged_text = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(df['text'])]
w2v_model = Doc2Vec(tagged_text, min_count=4, vector_size=100)


# In[43]:


textblob_text_vectors=[w2v_model.infer_vector(text) for text in df['clean_text']]


# ## Model Building

# In[44]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix


# In[45]:


models = {'RandomForestClassifier': RandomForestClassifier(n_estimators=100,max_depth=5),
          'MultinomialNB': MultinomialNB()}


# In[46]:


model_classifier=[]
Accuracy=[]
Precision=[]
Recall=[]
F1_SCORE=[]

for vectorizer,v in collection_df.items():
    for sentimenter,v in v.items():
        X=collection_df[vectorizer][sentimenter]['X']
        y=collection_df[vectorizer][sentimenter]['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in models.items():
            model_classifier.append(vectorizer+"+"+sentimenter+"+"+name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            Accuracy.append(round(accuracy_score(y_test, y_pred),4))
            Precision.append(round(precision_score(y_test, y_pred, average='weighted',zero_division=1),4))
            Recall.append(round(recall_score(y_test, y_pred, average='weighted'),4))
            F1_SCORE.append(round(f1_score(y_test, y_pred, average='weighted'),4))
            print("Done: "+vectorizer+"+"+sentimenter+"+"+name,"Accuracy: "+str(accuracy_score(y_test, y_pred)))


# In[ ]:




