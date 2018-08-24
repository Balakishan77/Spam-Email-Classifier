#Categorizing given email is spam or ham
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


dataset = pd.read_csv('spamham.csv')
dataset.head() 
dataset .columns #Index(['text', 'spam'], dtype='object')
dataset.shape  #(5728, 2)
#Checking for duplicates and removing them
dataset.drop_duplicates(inplace = True)
dataset.shape  #(5695, 2)
#Checking for any null entries in the dataset
print (pd.DataFrame(dataset.isnull().sum()))
#Checking class distribution
dataset.groupby('spam').count()
'''
spam      
0     4327
1     1368
'''
dataset['length'] = dataset['text'].map(lambda text: len(text))
#Let's plot histogram for length distribution by spam
dataset.hist(column='length', by='spam', bins=50)
#we can see some extreme outliers, we'll set a threshold for text length and plot the histogram again
dataset[dataset.length < 10000].hist(column='length', by='spam', bins=100)
#Using Natural Language Processing to cleaning the text to make one corpus
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Every mail starts with 'Subject :' lets remove this from each mail 
dataset['text']=dataset['text'].map(lambda text: text[1:])
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^A-Za-z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
ps = PorterStemmer()
corpus=dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))

'''
# Implemenation of corpus using function
corpus=[]
def fun(i):
    #return (list(filter(lambda text:text not in set(stopwords.words('english')),i)))
    return list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),i)))))
corpus= dataset['text'][0:5].apply(lambda i: fun(i))
'''
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus.values).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Fitting classifier to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#this function computes subset accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #0.99122036874451269
accuracy_score(y_test, y_pred,normalize=False) #1129 out of 1139

# Applying k-Fold Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)#array([ 0.98903509,  0.98903509,  0.99122807,  0.98026316,  0.98245614,0.98903509,  0.98901099,  0.99340659,  0.99340659,  0.98681319])
accuracies.mean()# 0.98836899942163114
accuracies.std()#0.0040467182445280397


# Create CV training and test scores for various training set sizes
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(classifier, X, y,cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


    
    
