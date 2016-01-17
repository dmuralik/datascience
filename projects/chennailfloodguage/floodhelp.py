
# coding: utf-8

# In[16]:

from twython import Twython
import re

APP_KEY = 'MC3NUUTUL4Qpp2sRgfAjSnFqB'
APP_SECRET = 'CEaH5Jz0I4ED8D3ZIK0n4uNu8ZjMfhoSIbziT0i1r06TwYx68R'
OAUTH_TOKEN = '18395318-YcHU9RsV8tSx3QXzxDYZ0FNpDpLLzbnyD99AjSOkh'
OAUTH_TOKEN_SECRET = '2YrKuyVPKYOY1lfhSfbBeiPqSa0epm5WMm2bfFxMi26qd'

twitter = Twython(APP_KEY, APP_SECRET,
                  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
#twitter.verify_credentials()
#results = twitter.search(q='#ChennaiRainsHelp')
#results = twitter.cursor(twitter.search, q='#ChennaiRainsHelp')
results = twitter.search(q='#ChennaiRainsHelp',   #**supply whatever query you want here**
                  count=100)
tweets = results['statuses']
searchForOffers = ['food', 'water', 'packet', 'blanket', 'packets', 'blankets']
searchForNeeds = ['need', 'want', 'wanted']
phoneNoPattern = re.compile('\d{10}')
for tweet in tweets:
    if any(word in tweet['text'].lower() for word in searchForNeeds):
        neededContact = phoneNoPattern.search(tweet['text'])
        if neededContact:
            print(neededContact.group() + '\n\n')
    elif any(word in tweet['text'].lower() for word in searchForOffers):
        offeredContact = phoneNoPattern.search(tweet['text'])
        if offeredContact:
            print(offeredContact.group() + '\n\n')
        


# In[ ]:



