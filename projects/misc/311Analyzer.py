
# coding: utf-8

# In[1]:

import pandas as pd

raw = pd.read_csv('/Users/dmurali/Downloads/nyc311calls.csv')

#print(raw['Unique Key'].values.size)
#print(raw.groupby('Agency').size().order(ascending=False))
#print(raw.groupby('Complaint Type').size().order()[:20])
#print(raw.groupby('Borough').size().order()[:20])
#print(raw.groupby(['Complaint Type','Borough']).size().order(ascending=False)[:50])
#print(raw[raw['Complaint Type']=='Unspecified'].groupby('Complaint Type').size().order())
wholeHour = [pd.DatetimeIndex(raw['Created Date']).hour if(mn==0) else -1 for mn in pd.DatetimeIndex(raw['Created Date']).minute]
raw['wholeHour'] = wholeHour
print(raw.groupby('wholeHour').size().order(ascending=False)[:5])
print(raw.groupby('wholeHour').size().order()[:5])


# In[ ]:



