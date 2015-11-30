
# coding: utf-8

# In[1]:

test


# In[2]:

print('test')


# In[12]:

import numpy as np
import pandas as pd
np.random.seed(12345)

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)

nums = np.random.rand(len(data))
mask_large = nums > 0.5

data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'
data.head()


# In[ ]:




# In[9]:

from sklearn.preprocessing import LabelEncoder
data = ['first', 'second', 'third', 'fourth']
enc = LabelEncoder()
label_encoder = enc.fit(data)
integer_classes = label_encoder.transform(label_encoder.classes_)
integer_classes


# In[12]:

from spam.spamhaus import SpamHausChecker
checker = SpamHausChecker()
checker.is_spam("http://www.google.com/search?q=food")


# In[16]:

from sklearn.ensemble import RandomForestClassifier
import numpy as np
print('test')


# In[12]:

import pandas as pd
import numpy as np
import seaborn as sns
df = pd.DataFrame({'a':[1,1,0,1], 'b' : [1,1,1,0]})
sum((df['a']==1) & (df['b']==1))


# In[17]:

import operator
stats = {'a':1000, 'b':3000, 'c': 100}
max(stats, key=stats.get)


# In[19]:

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

get_ipython().magic('matplotlib inline')
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# display the first 5 rows
#sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
linreg = LinearRegression()
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


# In[21]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.plot([1,2,3],[4,5,6])


# In[24]:

import pandas as pd
import numpy as np
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
  'foo', 'bar', 'foo', 'foo'],
  'B' : ['one', 'one', 'two', 'three',
  'two', 'two', 'one', 'three'],
               'C' : np.random.randn(8),
          'D' : np.random.randn(8)})
df.loc[df.B == 'one' or df.A == 'bar', 'A'] = 1
df


# In[38]:

import pandas as pd
import numpy as np
df = pd.DataFrame({'A':[1,0,1,1,0,1,1,0]})
b = pd.DataFrame([1,1,1,1,1,1], columns = ['B'])
final = pd.concat([df, b], axis = 1)
final['C'] = np.where(final['B'].isin(final['A']), 1, final['A'])
final


# In[70]:

import re

myString = " http://"

r = re.compile(r"(http://+)|(www+)")
res = r.search(myString)
if res is None :
    print('non')
else:
    print('not non')


# In[73]:

import pandas as pd

a = pd.DataFrame({'A' : ['A1','A2','A3']})
b = pd.DataFrame({'B' : ['B1','B2','B3']})
a['ab'] = a['A'] + ' ' + b['B']
a





# In[178]:

import numpy as np
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
text = ['poem to poem then was below']
text.append('going to be interesting cash')
tx = CountVectorizer().fit(text)
vect = tx.transform(text).toarray()
summed = np.sum(vect,axis = 0)
wordCount = sorted(zip(summed,tx.get_feature_names()), reverse = True)
#wordCount[0][1]
#print(sorted(wordCount, reverse = True)[:2])
#tfidf_transformer = TfidfTransformer().fit(vect)
#tfidf4 = tfidf_transformer.transform(vect)
df = pd.DataFrame(wordCount, columns = ['Count', 'Word'])

sb.barplot(x='Word', y='Count', data = df)


# In[2]:

from textblob import TextBlob
from nltk.corpus import stopwords
def splitIntoLemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

msg = 'was in there wanting doing'
splitIntoLemmas(msg)
stopwords.words("english")


# In[11]:

import pandas as pd
import numpy as np

df = pd.DataFrame({'A' : ['foo one', 'bar two', 'foo three', 'bar two',
  'foo', 'bar eight', 'foo', 'foo'],
               'C' : np.random.randn(8),
          'D' : np.random.randn(8)})
df[df['A'].str.contains('two')]


# In[32]:

from numpy import matrix
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([0.456, 0.3412544646464, 3.33, 4.6745, 5.54, 6.663])
m = csr_matrix((data, (row, col)), shape=(3, 3))
print(type(m))
print(type(m.toarray()))
print(m[0])
print(m.toarray()[0])


# In[49]:

import pandas as pd

df = pd.DataFrame({'A' : ['1', '2', '3', '4',
  '5', '6', '7', '8'],
               'C' : np.random.randn(8),
          'D' : np.random.randn(8)})
df['A'] = df['A'].astype(int)
df








# In[ ]:



