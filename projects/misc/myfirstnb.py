
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


# In[ ]:



