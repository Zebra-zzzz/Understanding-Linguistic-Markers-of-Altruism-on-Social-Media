
# coding: utf-8

# In[3]:


import codecs
import json

def read_dataset(path):
  with codecs.open(path, 'r', 'utf-8') as myFile:
    content = myFile.read()
  dataset = json.loads(content)
  return dataset

if __name__ == '__main__':
  path = 'pizza_request_dataset.json'
  dataset = read_dataset(path)


# In[6]:


print ('The dataset contains %d samples.' %(len(dataset)))


# In[7]:


print ('Available attributes: ', sorted(dataset[0].keys()))


# In[9]:


print ('First post:')
print (json.dumps(dataset[0], sort_keys=True, indent=2))


# In[10]:


successes = [r['requester_received_pizza'] for r in dataset]
successes


# In[12]:


success_rate = 100.0 * sum(successes) / float(len(successes))
print ('The average success rate is: %.2f%%' %(success_rate))


# # Model 1 – n-grams

# In[4]:


#Start Assignment 2

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
#from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


df = pd.read_json('pizza_request_dataset.json')
df.head()


# In[6]:


# Model 1 – n-grams

# Change all the text to lower case
df['request_text'] = [post.lower() for post in df['request_text']]
df.head()


# In[7]:


# Split the texts into tokens
df['request_text']= [nltk.word_tokenize(post) for post in df['request_text']]
df.head()


# In[44]:


# Remove Stop words, Non-alphabet and perfom Word Stemming/Lemmenting.


# In[8]:


for index,post in enumerate(df['request_text']):
    # Declaring Empty List to store the finalized words which will be the value of a new column 'review_final'
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
   
    for word in post:
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word)
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'review_final'
    df.loc[index,'review_final'] = str(Final_words)


# In[50]:


df.head()


# In[9]:


# Encode target label (pizza request succeed or fail)
Encoder = LabelEncoder()
df['requester_received_pizza'] = Encoder.fit_transform(df['requester_received_pizza'])


# In[14]:


# Split data into training and testing set
X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(df['review_final'], df['requester_received_pizza'], test_size=0.1, random_state=42)


# In[15]:


from nltk import bigrams
import re

#Join all training posts (the 'review_final' column) to train_words and then used Regex to capture all words across posts
train_words = ' '.join(X1_train)
train_words = re.findall('\'(\w+)\'', train_words)


# In[95]:


# Get unigrams
uni = sorted(set(train_words))
# Count the frequency of each unigram in the training posts
uni_count = [(word, train_words.count(word)) for word in uni]
uni_count


# In[104]:


# Sort uni_count based on the second element (the counted frequency) of each tuple
uni_count.sort(reverse=True, key = lambda x: x[1])
uni_count
# Get the top 500 unigrams
uni_final = [word[0] for word in uni_count[:500]]
uni_final


# In[122]:


# Construct a list that contains all of the bigrams across tranining posts
train_words_bi = []
# Get bigrams post by post
for l in X1_train:
    l_words = re.findall('\'(\w+)\'', l)
    l_bi = bigrams(l_words)
    train_words_bi.append(l_bi)
# Flatten the list
train_words_bi = [bi_words for l in train_words_bi for bi_words in l]
train_words_bi    


# In[126]:


print(len(train_words))
print(len(train_words_bi))


# In[129]:


# Get unique bigrams
bi = sorted(set(train_words_bi))
# Count the frequency of each bigram in the training posts
bi_count = [(words, train_words_bi.count(words)) for words in bi]
bi_count


# In[138]:


# Sort bi_count based on the second element (the counted frequency) of each tuple
bi_count.sort(reverse=True, key = lambda x: x[1])
bi_count
# Get the top 500 unigrams
bi_final = [words[0] for words in bi_count[:500]]
bi_final = [words[0] + ' ' + words[1] for words in bi_final]
bi_final


# In[112]:


# Fit the top 500 unigrams to the CountVectorizer
vect = CountVectorizer().fit(uni_final)
vect.get_feature_names()


# In[143]:


# Fit the top 500 bigrams to the CountVectorizer
vect2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)).fit(bi_final)
vect2.get_feature_names()


# In[156]:


# Transform the training and test data to a matrix respectively for post vectorization based on unigrams
X1_uni_train_vectorized = vect.transform(X1_train)
X1_uni_test_vectorized = vect.transform(X1_test)

print(X1_uni_train_vectorized.toarray())


# In[157]:


# Transform the training and test data to a matrix respectively for post vectorization based on bigrams
X1_bi_train_vectorized = vect2.transform(X1_train)
X1_bi_test_vectorized = vect2.transform(X1_test)

print(X1_bi_train_vectorized.toarray())


# In[160]:


# Join the unigrams matrix and bigrams matrix to a matrix (n*1000 dimensions) 
import scipy.sparse as sp
X1_train_vectorized = sp.hstack((X1_uni_train_vectorized, X1_bi_train_vectorized), format='csr')
X1_test_vectorized = sp.hstack((X1_uni_test_vectorized, X1_bi_test_vectorized), format='csr')


# In[262]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM1 = svm.SVC(C=1.0, kernel='linear', class_weight = 'balanced')
SVM1.fit(X1_train_vectorized, y1_train)
# predict the labels on validation dataset
predictions_SVM1 = SVM1.predict(X1_test_vectorized)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y1_test, predictions_SVM1)*100)


# In[263]:


#from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
print('Accuracy Score: ', accuracy_score(y1_test, predictions_SVM1))
print('Precision Score: ', precision_score(y1_test, predictions_SVM1))
print('Recall Score: ', recall_score(y1_test, predictions_SVM1))
print('F1 Score: ', f1_score(y1_test, predictions_SVM1))
#print('AUC Score: ', metrics.roc_auc_score(y1_test, predictions_SVM1))
fpr, tpr, thresholds = roc_curve(y1_test, predictions_SVM1)
print('AUC Score: ', auc(fpr, tpr))
tn, fp, fn, tp = confusion_matrix(y1_test, predictions_SVM1).ravel()
print('Specificity: ', tn / (tn+fp))


# # Model 2 – Activity and Reputation

# In[136]:


# Model 2 – Activity and Reputation


df['post_was_edited'] = Encoder.fit_transform(df['post_was_edited'])

# Split data into training and testing set
X2_1_train, X2_1_test, y2_train, y2_test = model_selection.train_test_split(df['requester_subreddits_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
#X2_2_train, X2_2_test, y2_train, y2_test = model_selection.train_test_split(df['requester_user_flair'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_3_train, X2_3_test, y2_train, y2_test = model_selection.train_test_split(df['post_was_edited'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_4_train, X2_4_test, y2_train, y2_test = model_selection.train_test_split(df['number_of_downvotes_of_request_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_5_train, X2_5_test, y2_train, y2_test = model_selection.train_test_split(df['number_of_upvotes_of_request_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_6_train, X2_6_test, y2_train, y2_test = model_selection.train_test_split(df['requester_upvotes_plus_downvotes_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_7_train, X2_7_test, y2_train, y2_test = model_selection.train_test_split(df['requester_account_age_in_days_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_8_train, X2_8_test, y2_train, y2_test = model_selection.train_test_split(df['requester_account_age_in_days_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_9_train, X2_9_test, y2_train, y2_test = model_selection.train_test_split(df['requester_days_since_first_post_on_raop_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_10_train, X2_10_test, y2_train, y2_test = model_selection.train_test_split(df['requester_days_since_first_post_on_raop_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_11_train, X2_11_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_comments_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_12_train, X2_12_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_comments_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_13_train, X2_13_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_comments_in_raop_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_14_train, X2_14_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_comments_in_raop_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_15_train, X2_15_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_posts_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_16_train, X2_16_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_posts_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_17_train, X2_17_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_posts_on_raop_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_18_train, X2_18_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_posts_on_raop_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_19_train, X2_19_test, y2_train, y2_test = model_selection.train_test_split(df['requester_number_of_subreddits_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_20_train, X2_20_test, y2_train, y2_test = model_selection.train_test_split(df['requester_upvotes_minus_downvotes_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_21_train, X2_21_test, y2_train, y2_test = model_selection.train_test_split(df['requester_upvotes_minus_downvotes_at_retrieval'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X2_22_train, X2_22_test, y2_train, y2_test = model_selection.train_test_split(df['requester_upvotes_plus_downvotes_at_request'], df['requester_received_pizza'], test_size=0.1, random_state=42)


y2_test == y1_test


# In[137]:


# Use unigram to vectorize 'requester_subreddits_at_request'

# Constrcut a list that contains the string of each post's 'requester_subreddits_at_request'
X2_1_train = [str(l) for l in X2_1_train]
X2_1_test = [str(l) for l in X2_1_test]

# Convert all subreddits from training posts to feature names
vect_subr_uni = CountVectorizer().fit(X2_1_train)
vect_subr_uni.get_feature_names()



# In[138]:


# Transform the training and test data to a matrix respectively for 'requester_subreddits_at_request' vectorization based on subreddits
X2_subr_train_vectorized = vect_subr_uni.transform(X2_1_train)
X2_subr_test_vectorized = vect_subr_uni.transform(X2_1_test)

#print(X2_subr_train_vectorized.toarray())


# In[139]:


# Convert data frames to matrices
X2_3_train_vectorized = sp.csr_matrix(X2_3_train).transpose()
X2_4_train_vectorized = sp.csr_matrix(X2_4_train).transpose()
X2_5_train_vectorized = sp.csr_matrix(X2_5_train).transpose()
X2_6_train_vectorized = sp.csr_matrix(X2_6_train).transpose()
X2_7_train_vectorized = sp.csr_matrix(X2_7_train).transpose()
X2_8_train_vectorized = sp.csr_matrix(X2_8_train).transpose()
X2_9_train_vectorized = sp.csr_matrix(X2_9_train).transpose()
X2_10_train_vectorized = sp.csr_matrix(X2_10_train).transpose()
X2_11_train_vectorized = sp.csr_matrix(X2_11_train).transpose()
X2_12_train_vectorized = sp.csr_matrix(X2_12_train).transpose()
X2_13_train_vectorized = sp.csr_matrix(X2_13_train).transpose()
X2_14_train_vectorized = sp.csr_matrix(X2_14_train).transpose()
X2_15_train_vectorized = sp.csr_matrix(X2_15_train).transpose()
X2_16_train_vectorized = sp.csr_matrix(X2_16_train).transpose()
X2_17_train_vectorized = sp.csr_matrix(X2_17_train).transpose()
X2_18_train_vectorized = sp.csr_matrix(X2_18_train).transpose()
X2_19_train_vectorized = sp.csr_matrix(X2_19_train).transpose()
X2_20_train_vectorized = sp.csr_matrix(X2_20_train).transpose()
X2_21_train_vectorized = sp.csr_matrix(X2_21_train).transpose()
X2_22_train_vectorized = sp.csr_matrix(X2_22_train).transpose()

X2_3_test_vectorized = sp.csr_matrix(X2_3_test).transpose()
X2_4_test_vectorized = sp.csr_matrix(X2_4_test).transpose()
X2_5_test_vectorized = sp.csr_matrix(X2_5_test).transpose()
X2_6_test_vectorized = sp.csr_matrix(X2_6_test).transpose()
X2_7_test_vectorized = sp.csr_matrix(X2_7_test).transpose()
X2_8_test_vectorized = sp.csr_matrix(X2_8_test).transpose()
X2_9_test_vectorized = sp.csr_matrix(X2_9_test).transpose()
X2_10_test_vectorized = sp.csr_matrix(X2_10_test).transpose()
X2_11_test_vectorized = sp.csr_matrix(X2_11_test).transpose()
X2_12_test_vectorized = sp.csr_matrix(X2_12_test).transpose()
X2_13_test_vectorized = sp.csr_matrix(X2_13_test).transpose()
X2_14_test_vectorized = sp.csr_matrix(X2_14_test).transpose()
X2_15_test_vectorized = sp.csr_matrix(X2_15_test).transpose()
X2_16_test_vectorized = sp.csr_matrix(X2_16_test).transpose()
X2_17_test_vectorized = sp.csr_matrix(X2_17_test).transpose()
X2_18_test_vectorized = sp.csr_matrix(X2_18_test).transpose()
X2_19_test_vectorized = sp.csr_matrix(X2_19_test).transpose()
X2_20_test_vectorized = sp.csr_matrix(X2_20_test).transpose()
X2_21_test_vectorized = sp.csr_matrix(X2_21_test).transpose()
X2_22_test_vectorized = sp.csr_matrix(X2_22_test).transpose()

# Join each matrix to a single matrix
X2_train_vectorized = sp.hstack((X2_subr_train_vectorized, X2_3_train_vectorized, X2_4_train_vectorized, X2_5_train_vectorized, X2_6_train_vectorized, X2_7_train_vectorized, X2_8_train_vectorized, X2_9_train_vectorized, X2_10_train_vectorized, X2_11_train_vectorized, X2_12_train_vectorized, X2_13_train_vectorized, X2_14_train_vectorized, X2_15_train_vectorized, X2_16_train_vectorized, X2_17_train_vectorized, X2_18_train_vectorized, X2_19_train_vectorized, X2_20_train_vectorized, X2_21_train_vectorized, X2_22_train_vectorized), format='csr')
X2_test_vectorized = sp.hstack((X2_subr_test_vectorized, X2_3_test_vectorized, X2_4_test_vectorized, X2_5_test_vectorized, X2_6_test_vectorized, X2_7_test_vectorized, X2_8_test_vectorized, X2_9_test_vectorized, X2_10_test_vectorized, X2_11_test_vectorized, X2_12_test_vectorized, X2_13_test_vectorized, X2_14_test_vectorized, X2_15_test_vectorized, X2_16_test_vectorized, X2_17_test_vectorized, X2_18_test_vectorized, X2_19_test_vectorized, X2_20_test_vectorized, X2_21_test_vectorized, X2_22_test_vectorized), format='csr')
print(X2_train_vectorized.toarray())


# In[140]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM2 = svm.SVC(C=1.0, kernel='linear', class_weight = 'balanced')
SVM2.fit(X2_train_vectorized, y2_train)
# predict the labels on validation dataset
predictions_SVM2 = SVM2.predict(X2_test_vectorized)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y2_test, predictions_SVM2)*100)


# In[141]:


#from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
print('Accuracy Score: ', accuracy_score(y2_test, predictions_SVM2))
print('Precision Score: ', precision_score(y2_test, predictions_SVM2))
print('Recall Score: ', recall_score(y2_test, predictions_SVM2))
print('F1 Score: ', f1_score(y2_test, predictions_SVM2))
#print('AUC Score: ', metrics.roc_auc_score(y2_test, predictions_SVM))
fpr, tpr, thresholds = roc_curve(y2_test, predictions_SVM2)
print('AUC Score: ', auc(fpr, tpr))
tn, fp, fn, tp = confusion_matrix(y2_test, predictions_SVM2).ravel()
print('Specificity: ', tn / (tn+fp))


# # Model 3 – Narratives

# In[26]:


# Model 3 – Narratives

# Use join to get the regex expression for narrative_desire, family, job, money, and student
fhand_desire = open('narratives/desire.txt')
text_desire = fhand_desire.read()
rg_desire = '|'.join(text_desire.split('\n'))

print(rg_desire)

fhand_family = open('narratives/family.txt')
text_family= fhand_family.read()
rg_family = '|'.join(text_family.split('\n'))

fhand_job = open('narratives/job.txt')
text_job= fhand_job.read()
rg_job = '|'.join(text_job.split('\n'))

fhand_money = open('narratives/money.txt')
text_money= fhand_money.read()
rg_money = '|'.join(text_money.split('\n'))

fhand_student = open('narratives/student.txt')
text_student= fhand_student.read()
rg_student = '|'.join(text_student.split('\n'))

# Perform Regex match. Calculate the ratio of the number of matches for each narrative to the total number of white spaced words in the post, and store in new columns
for index, post in enumerate(df['review_final']):
    total = len(re.findall('\', \'', post)) + 1
    df.loc[index, 'narra_desire'] = len(re.findall(rg_desire, post))/total
    df.loc[index, 'narra_family'] = len(re.findall(rg_family, post))/total
    df.loc[index, 'narra_job'] = len(re.findall(rg_job, post))/total
    df.loc[index, 'narra_money'] = len(re.findall(rg_money, post))/total
    df.loc[index, 'narra_student'] = len(re.findall(rg_student, post))/total


# In[27]:


df.head()


# In[28]:


# Split data into training and testing set
X3_1_train, X3_1_test, y3_train, y3_test = model_selection.train_test_split(df['narra_desire'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X3_2_train, X3_2_test, y3_train, y3_test = model_selection.train_test_split(df['narra_family'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X3_3_train, X3_3_test, y3_train, y3_test = model_selection.train_test_split(df['narra_job'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X3_4_train, X3_4_test, y3_train, y3_test = model_selection.train_test_split(df['narra_money'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X3_5_train, X3_5_test, y3_train, y3_test = model_selection.train_test_split(df['narra_student'], df['requester_received_pizza'], test_size=0.1, random_state=42)


# In[29]:


# Convert data frames to matrices
X3_1_train_vectorized = sp.csr_matrix(X3_1_train).transpose()
X3_2_train_vectorized = sp.csr_matrix(X3_2_train).transpose()
X3_3_train_vectorized = sp.csr_matrix(X3_3_train).transpose()
X3_4_train_vectorized = sp.csr_matrix(X3_4_train).transpose()
X3_5_train_vectorized = sp.csr_matrix(X3_5_train).transpose()

X3_1_test_vectorized = sp.csr_matrix(X3_1_test).transpose()
X3_2_test_vectorized = sp.csr_matrix(X3_2_test).transpose()
X3_3_test_vectorized = sp.csr_matrix(X3_3_test).transpose()
X3_4_test_vectorized = sp.csr_matrix(X3_4_test).transpose()
X3_5_test_vectorized = sp.csr_matrix(X3_5_test).transpose()

# Join each matrix to a single matrix
X3_train_vectorized = sp.hstack((X3_1_train_vectorized, X3_2_train_vectorized, X3_3_train_vectorized, X3_4_train_vectorized, X3_5_train_vectorized), format='csr')
X3_test_vectorized = sp.hstack((X3_1_test_vectorized, X3_2_test_vectorized, X3_3_test_vectorized, X3_4_test_vectorized, X3_5_test_vectorized), format='csr')

print(X3_train_vectorized.toarray())


# In[30]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM3 = svm.SVC(C=1.0, kernel='linear', class_weight = 'balanced')
SVM3.fit(X3_train_vectorized, y3_train)
# predict the labels on validation dataset
predictions_SVM3 = SVM3.predict(X3_test_vectorized)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y3_test, predictions_SVM3)*100)


# In[31]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
print('Accuracy Score: ', accuracy_score(y3_test, predictions_SVM3))
print('Precision Score: ', precision_score(y3_test, predictions_SVM3))
print('Recall Score: ', recall_score(y3_test, predictions_SVM3))
print('F1 Score: ', f1_score(y3_test, predictions_SVM3))
fpr, tpr, thresholds = roc_curve(y3_test, predictions_SVM3)
print('AUC Score: ', auc(fpr, tpr))
tn, fp, fn, tp = confusion_matrix(y3_test, predictions_SVM3).ravel()
print('Specificity: ', tn / (tn+fp))


# In[17]:


import scipy.sparse as sp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# # Model 4 – Moral foundations

# In[32]:


# Model 4 – Moral foundations

# Construct a list that contains each line of the moral dict. Use strip() to remove the white spaces at the beginning and end of line
fhand_moral = open('MoralFoundations.dic')
moral_dict = [row.strip() for row in fhand_moral]
moral_dict


# In[33]:


# Calculate where (which line) does the dict actually start
i = 0
for line in moral_dict:
    if line.startswith('safe'):
        print(i)
        break
    i = i+1  


# In[34]:


# Slicing the dict to only contain words without index
moral_dict = moral_dict[i:]
moral_dict


# In[35]:


# Construct vocabulary lists for each moral foundation dimension
HarmVirtue_dict = []
HarmVice_dict = []
IngroupVirtue_dict = []
IngroupVice_dict = []
AuthorityVirtue_dict = []
AuthorityVice_dict = []
PurityVirtue_dict = []
PurityVice_dict = []


# In[36]:


for line in moral_dict:
    if re.search('01', line):  
        HarmVirtue_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*01', line)[0])
    if re.search('02', line):  
        HarmVice_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*02', line)[0])
    if re.search('05', line):  
        IngroupVirtue_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*05', line)[0])   
    if re.search('06', line):  
        IngroupVice_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*06', line)[0])   
    if re.search('07', line):  
        AuthorityVirtue_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*07', line)[0])   
    if re.search('08', line):  
        AuthorityVice_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*08', line)[0])   
    if re.search('09', line):  
        PurityVirtue_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*09', line)[0])   
    if re.search('10', line):  
        PurityVice_dict.append(re.findall('([a-z]+)\*?\s+(?:\d* )*10', line)[0])   

print(PurityVice_dict)


# In[119]:


# Use join to get the regex expression for each moral foundation dimension
rg_HarmVirtue = '|'.join(HarmVirtue_dict)
rg_HarmVice = '|'.join(HarmVice_dict)
rg_IngroupVirtue = '|'.join(IngroupVirtue_dict)
rg_IngroupVice = '|'.join(IngroupVice_dict)
rg_AuthorityVirtue = '|'.join(AuthorityVirtue_dict)
rg_AuthorityVice = '|'.join(AuthorityVice_dict)
rg_PurityVirtue = '|'.join(PurityVirtue_dict)
rg_PurityVice = '|'.join(PurityVice_dict)

rg_PurityVice


# In[121]:


# Perform Regex match. Calculate the ratio of the number of matches for each moral foundation to the total number of white spaced words in the post, and store in new columns
for index, post in enumerate(df['review_final']):
    total = len(re.findall('\', \'', post)) + 1
    df.loc[index, 'moral_harm+'] = len(re.findall(rg_HarmVirtue, post))/total
    df.loc[index, 'moral_harm-'] = len(re.findall(rg_HarmVice, post))/total
    df.loc[index, 'moral_ingroup+'] = len(re.findall(rg_IngroupVirtue, post))/total
    df.loc[index, 'moral_ingroup-'] = len(re.findall(rg_IngroupVice, post))/total
    df.loc[index, 'moral_authority+'] = len(re.findall(rg_AuthorityVirtue, post))/total
    df.loc[index, 'moral_authority-'] = len(re.findall(rg_AuthorityVice, post))/total
    df.loc[index, 'moral_purity+'] = len(re.findall(rg_PurityVirtue, post))/total
    df.loc[index, 'moral_purity-'] = len(re.findall(rg_PurityVice, post))/total


# In[122]:


df.head()


# In[123]:


# Split data into training and testing set
X4_1_train, X4_1_test, y4_train, y4_test = model_selection.train_test_split(df['moral_harm+'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_2_train, X4_2_test, y4_train, y4_test = model_selection.train_test_split(df['moral_harm-'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_3_train, X4_3_test, y4_train, y4_test = model_selection.train_test_split(df['moral_ingroup+'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_4_train, X4_4_test, y4_train, y4_test = model_selection.train_test_split(df['moral_ingroup-'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_5_train, X4_5_test, y4_train, y4_test = model_selection.train_test_split(df['moral_authority+'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_6_train, X4_6_test, y4_train, y4_test = model_selection.train_test_split(df['moral_authority-'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_7_train, X4_7_test, y4_train, y4_test = model_selection.train_test_split(df['moral_purity+'], df['requester_received_pizza'], test_size=0.1, random_state=42)
X4_8_train, X4_8_test, y4_train, y4_test = model_selection.train_test_split(df['moral_purity-'], df['requester_received_pizza'], test_size=0.1, random_state=42)


# In[124]:


# Convert data frames to matrices
X4_1_train_vectorized = sp.csr_matrix(X4_1_train).transpose()
X4_2_train_vectorized = sp.csr_matrix(X4_2_train).transpose()
X4_3_train_vectorized = sp.csr_matrix(X4_3_train).transpose()
X4_4_train_vectorized = sp.csr_matrix(X4_4_train).transpose()
X4_5_train_vectorized = sp.csr_matrix(X4_5_train).transpose()
X4_6_train_vectorized = sp.csr_matrix(X4_6_train).transpose()
X4_7_train_vectorized = sp.csr_matrix(X4_7_train).transpose()
X4_8_train_vectorized = sp.csr_matrix(X4_8_train).transpose()

X4_1_test_vectorized = sp.csr_matrix(X4_1_test).transpose()
X4_2_test_vectorized = sp.csr_matrix(X4_2_test).transpose()
X4_3_test_vectorized = sp.csr_matrix(X4_3_test).transpose()
X4_4_test_vectorized = sp.csr_matrix(X4_4_test).transpose()
X4_5_test_vectorized = sp.csr_matrix(X4_5_test).transpose()
X4_6_test_vectorized = sp.csr_matrix(X4_6_test).transpose()
X4_7_test_vectorized = sp.csr_matrix(X4_7_test).transpose()
X4_8_test_vectorized = sp.csr_matrix(X4_8_test).transpose()

# Join each matrix to a single matrix
X4_train_vectorized = sp.hstack((X4_1_train_vectorized, X4_2_train_vectorized, X4_3_train_vectorized, X4_4_train_vectorized, X4_5_train_vectorized, X4_6_train_vectorized, X4_7_train_vectorized, X4_8_train_vectorized), format='csr')
X4_test_vectorized = sp.hstack((X4_1_test_vectorized, X4_2_test_vectorized, X4_3_test_vectorized, X4_4_test_vectorized, X4_5_test_vectorized, X4_6_test_vectorized, X4_7_test_vectorized, X4_8_test_vectorized), format='csr')

print(X4_train_vectorized.toarray())


# In[133]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM4 = svm.SVC(C=1.0, kernel='linear', class_weight = 'balanced')
SVM4.fit(X4_train_vectorized, y4_train)
# predict the labels on validation dataset
predictions_SVM4 = SVM4.predict(X4_test_vectorized)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y4_test, predictions_SVM4)*100)


# In[135]:


#from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
print('Accuracy Score: ', accuracy_score(y4_test, predictions_SVM4))
print('Precision Score: ', precision_score(y4_test, predictions_SVM4))
print('Recall Score: ', recall_score(y4_test, predictions_SVM4))
print('F1 Score: ', f1_score(y4_test, predictions_SVM4))
fpr, tpr, thresholds = roc_curve(y4_test, predictions_SVM4)
print('AUC Score: ', auc(fpr, tpr))
tn, fp, fn, tp = confusion_matrix(y4_test, predictions_SVM4).ravel()
print('Specificity: ', tn / (tn+fp))

