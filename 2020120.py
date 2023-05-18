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


# In[4]:


df.head()


# In[12]:


df['user_followers'] = pd.to_numeric(df['user_followers'], errors='coerce')

# Just converts all str types in the column to int so that comparison operator can be used


# In[53]:


df = df.drop_duplicates(keep='first')

# Removes duplicate rows as per instructions


# #### Checking Dataset

# In[14]:


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


# ##### Percentages of missing values per column

# In[15]:


(df.isnull().sum() / len(df)) * 100


# In[56]:


df = df.drop(['user_location', 'hashtags', 'user_description'], axis=1)

# Dropping rows with missing values as per instructions


# #### Filling NaN Values
df = df.dropna() # Not needed yetdf.head(2)
# #### Checking some features

# In[16]:


print(len(df[df['user_followers'] < 1000]), 'users with less than 1000 followers')
print(len(df[df['user_followers'] > 1000]), 'users with more than 1000 followers')


# ### Cleaning functions

# In[17]:


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


# In[18]:


# This was so that TQDM does not give AttributeError

df["text"] = df["text"].astype(str)
df["text"] = [x.replace(':',' ') for x in df["text"]]


# ### Running text cleaner and storing in new column

# In[19]:


df['clean_text'] = pd.Series([clean_text(i) for i in tqdm(df['text'])])


# In[20]:


words = df["clean_text"].values


# In[21]:


ls = []

for i in words:
    ls.append(str(i))


# In[22]:


ls[:5]


# ### Wordcloud 

# In[23]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="lightblue", colormap='Set2', max_words=1000, max_font_size= 200,  width=1600, height=800)
wc.generate(" ".join(ls))
plt.title("Most discussed terms", fontsize=20)
plt.imshow(wc.recolor( colormap= 'Set2' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')


# Trying some Features

# In[24]:


most_pop = df.sort_values('user_followers', ascending =False)[['user_name', 'user_followers']].head(12)

most_pop['user_followers1'] = most_pop['user_followers']/1000


# In[25]:


most_pop


# In[26]:


plt.figure(figsize = (20,25))

sns.barplot(data = most_pop, y = 'user_name', x = 'user_followers1', color = 'c')
plt.xticks(fontsize=27, rotation=0)
plt.yticks(fontsize=30, rotation=0)
plt.xlabel('User followers in Thousands', fontsize = 21)
plt.ylabel('')
plt.title('Followers', fontsize = 30);


# # TF/IDF

# In[44]:


from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


# In[45]:


lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics=10,
                                       id2word=dictionary,
                                       passes=2,
                                       workers=2)


# # Showing the output of the model

# In[46]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[47]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
                                             num_topics=10,
                                             id2word=dictionary,
                                             passes=2,
                                             workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[48]:


#39049th row, 2nd column 

df.iloc[234,1]


# In[49]:


unseen_document = 'so happy for the chatGPT team for com8ng up with such a revolutionary idea.The FUTURE LOOKS BRIGHT.'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

