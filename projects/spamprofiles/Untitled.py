
# coding: utf-8

# In[11]:

from MongoClient import read_mongo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
import time

print('test')
rawSite = read_mongo(db = 'CB', collection = 'site', host = 'localhost', no_id = False)
siteModified = rawSite.drop(['dismissedOnboarding', 'feedCounter', 'feedToken',
              'modules', 'password', 'theme', 'photoId', 'requestAccess',
              'requestPassword', 'cm', 'bi', 'photo', 'goFundMe', 'lastName',
              'numAmps', 'partner', 'size', 'theme', 'createFormSessionId', 'allowList',
              'blockList', 'displayEmail', 'isPhotoOrderingFixed', 'healthCondition',
              'spam', 'status', 'firstName', 'lastInvite'], axis = 1)
siteModified['descriptionLen'] = rawSite.description.str.len()
siteModified.drop(['description'], axis = 1, inplace = True)
siteModified['nameLen'] = rawSite.name.str.len()
siteModified.drop(['name'], axis = 1, inplace = True)
siteModified['titleLen'] = rawSite.title.str.len()
siteModified.drop(['title'], axis = 1, inplace = True)

siteCreatedDayOrNight = ['Night' if(hr > 22 or hr <= 6) else 'Day' for hr in pd.DatetimeIndex(siteModified['createdAt']).hour]
siteModified['siteCreatedDayOrNight'] = siteCreatedDayOrNight
siteModified.drop(['createdAt'], axis = 1, inplace = True)
siteUpdatedDayOrNight = ['Night' if(hr > 22 or hr <= 6) else 'Day' for hr in pd.DatetimeIndex(siteModified['updatedAt']).hour]
siteModified['siteUpdatedDayOrNight'] = siteUpdatedDayOrNight
siteModified.drop(['updatedAt'], axis = 1, inplace = True)

siteModified.descriptionLen.replace(np.nan, -1, inplace = True)
siteModified.age.replace(np.nan, 'blank', inplace = True)
siteModified.hasCommentFix.replace(np.nan, 0, inplace = True)
siteModified.hasVisitorInvite.replace(np.nan, 0, inplace = True)
siteModified.isDeleted.replace(np.nan, 0, inplace = True)
siteModified.isForSelf.replace(np.nan, 0, inplace = True)
siteModified.isSearchable.replace(np.nan, 0, inplace = True)
siteModified.isSpam.replace(np.nan, 0, inplace = True)
siteModified.sawReCaptcha.replace(np.nan, 0, inplace = True)

siteModified.rename(columns={'_id': 'siteId', 'isDeleted' : 'isSiteDeleted', 'sawReCaptcha' : 'sawReCaptchaSite', 'isSpam' : 'isSiteSpam'}, inplace=True)

binarizedSites = pd.get_dummies(siteModified, columns = ['platform', 'privacy', 'age'])

rawProfile = read_mongo(db = 'CB', collection = 'profile', host = 'localhost', no_id = False)
profileModified = rawProfile.drop(['ampProfile', 'bio', 'cm', 'country', 'createFormSessionId', 'employer',
                'feedCounter', 'guid', 'howFoundOther', 'language', 'lastDrawingTool',
                'lastLastLogin', 'lastLogin', 'lastModifier', 'lastName', 'firstName',
                'lastVideoUpload', 'location', 'mailingAddress', 'mobile', 'my', 'n',
                'notes', 'password', 'phone', 'photo', 'platform', 'sms', 'social',
                'tz', 'whitelistedByCustomerCare', 'handle', 'createdAt', 'updatedAt',
                'lastActivity', 'lastJournalReply', 'ip', 'howFound', 'isStub', 'failedLoginAttempts',
                'isMailSubscriber', 'spam', 'email', 'isSecure', 'isPrivate', 'isPublic',
                'gender'], axis = 1)
#emails = profileModified.email.apply(pd.Series)
#emailDomains = pd.DataFrame(emails.domain.values, columns = ['emailDomain'])
#withoutEmail = profileModified.drop(['email'], axis = 1)
#withDomains = pd.concat([withoutEmail, emailDomains], axis = 1).fillna('blank')
profileModified.sawReCaptcha.replace(np.nan, 0, inplace = True)
profileModified.isDeleted.replace(np.nan, 0, inplace = True)
profileModified.numNotifications.replace(np.nan, -1, inplace = True)

profileModified.rename(columns={'_id': 'profileId', 'isDeleted' : 'isProfileDeleted', 'sawReCaptcha' : 'sawReCaptchaProfile'}, inplace=True)


#read site_profile
rawSiteProfile = read_mongo(db = 'CB', collection = 'site_profile', host = 'localhost')
siteProfile = pd.DataFrame(rawSiteProfile['siteId'], columns = ['siteId'])
siteProfile['profileId'] = rawSiteProfile.userId

#read site_profile with spam
octSiteProfileSpam = pd.read_csv("/Users/dmurali/Documents/spamlist_round25_from_20150809_to_20151015.csv",
                    usecols = ['siteId','isSpam'])
octSiteProfileSpam.rename(columns = {'isSpam':'isOctSpam'}, inplace = True)

mergedSiteProfile = binarizedSites.merge(siteProfile, how='left', on = ['siteId'], sort = False).merge(profileModified, how='left', on = ['profileId'], sort = False).merge(octSiteProfileSpam, how='left', on = ['siteId'], sort = False)
mergedSiteProfile['isSpam'] = np.where(mergedSiteProfile['isOctSpam'].isin(mergedSiteProfile['isSiteSpam']), 1, mergedSiteProfile['isSiteSpam'])

mergedSiteProfile = mergedSiteProfile.convert_objects(convert_numeric=True)

isSpamTest = mergedSiteProfile.loc[mergedSiteProfile['isSpam'] == 1][:340]
isNotSpamTest = mergedSiteProfile.loc[mergedSiteProfile['isSpam'] == 0][:6300]
test = pd.concat([isSpamTest, isNotSpamTest])
train = mergedSiteProfile[~mergedSiteProfile.siteId.isin(test.siteId)]

train.drop(['siteId', 'profileId'], axis = 1, inplace = True)
test.drop(['siteId', 'profileId'], axis = 1, inplace = True)

#using randomforest
numTrees = range(10, 100, 10)
numMinLeafSamples = range(2, 20, 2)
numMinSamplesSplit = range(1, 20, 3)
param_dist = dict(n_estimators = numTrees, min_samples_leaf = numMinLeafSamples, min_samples_split = numMinSamplesSplit)
model = RandomForestClassifier(n_estimators=60)

rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
get_ipython().magic("time rand.fit(train, train['isSpam'])")
# examine the best model
print(rand.best_score_)
print(rand.best_params_)


# In[ ]:



