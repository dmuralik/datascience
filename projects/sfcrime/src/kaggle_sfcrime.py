
# coding: utf-8

# In[1]:

#This is for a Kaggle datascience competition. More information about it can be found here: https://www.kaggle.com/c/sf-crime/data
#I have used a RandomForest classifier for prediction.
import pandas as pd
import patsy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
from sklearn import cross_validation
from sklearn.metrics import log_loss
import logloss
import sys
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

#to track the performance
startTime = time.time()

#a function that can mungle the data (for reusability)
def wrangle(dataFrame, isTraining):
    dayOfWeekChanged = pd.get_dummies(dataFrame['DayOfWeek'])
    pdDistrictChanged = pd.get_dummies(dataFrame['PdDistrict'])
    dayOrNight = ['Night' if(hr > 22 or hr <= 6) else 'Day' for hr in pd.DatetimeIndex(dataFrame['Dates']).hour]
    dayOrNightChanged = pd.get_dummies(dayOrNight)
    month = pd.DatetimeIndex(dataFrame['Dates']).month
    monthFrame = pd.DataFrame(month, columns = ['Month'])
    year = pd.DatetimeIndex(dataFrame['Dates']).year
    yearFrame = pd.DataFrame(year, columns = ['Year'])
    enumIndex = categoryLabels = []
    dropped = dataFrame.drop(['Dates','DayOfWeek', 'PdDistrict'], axis=1)
    interimFrame = pd.concat([dropped,dayOfWeekChanged, pdDistrictChanged, dayOrNightChanged, monthFrame, yearFrame], axis=1)
    if(isTraining):
        enumIndex,categoryLabels = pd.factorize(interimFrame['Category'])
        categoryChanged = pd.DataFrame(enumIndex, columns = ['Category'])
        categoryDropped = interimFrame.drop(['Category'], axis = 1)
        transformed = pd.concat([categoryDropped, categoryChanged], axis=1)
    else:
        category = pd.DataFrame([], columns = ['Category'])
        transformed = pd.concat([interimFrame, category], axis=1)
        transformed['Category'].fillna(9999, inplace=True)
    return (transformed, enumIndex, categoryLabels)

#unindexes class labels to descriptions
def unIndex(labels, enumIndex):
    return [labels[index] for index in enumIndex]

def findMissingLabels(masterLabels, predictedLabels):
    return masterLabels - set(predictedLabels)

train = pd.read_csv("../data/train.csv",
                    usecols = ['Dates','Category','DayOfWeek','PdDistrict','X','Y'],
                    parse_dates = [1]
                    )
test = pd.read_csv("../data/test.csv",
                   usecols = ['Dates','DayOfWeek','PdDistrict','X','Y'],
                   parse_dates = [1]
                   )

trainingWrangled, enumIndexTraining, categoryLabelsTraining = wrangle(train, True)
testWrangled, enumIndexTest, categoryLabelsTest = wrangle(test, False)

#model the data using randomforest
#numTrees = range(10, 100, 10)
#numMinLeafSamples = range(2, 20, 2)
#numMinSamplesSplit = range(1, 20, 3)
#param_dist = dict(n_estimators = numTrees, min_samples_leaf = numMinLeafSamples, min_samples_split = numMinSamplesSplit)
#model = RandomForestClassifier(n_estimators=60)


# model the data using knn
# define the parameter values that should be searched
k_range = range(1, 50)
weight_options = ['uniform', 'distance']
# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)
model = KNeighborsClassifier(n_neighbors=24, weights='distance')

#model data using logistic regression
#model = LogisticRegression()
#%time print(np.sqrt(-cross_val_score(model, trainingWrangled, enumIndexTraining, cv=6, scoring='mean_squared_error')).mean())

rand = RandomizedSearchCV(model, param_dist, cv=6, scoring='accuracy', n_iter=8)
get_ipython().magic('time rand.fit(trainingWrangled, enumIndexTraining)')
# examine the best model
print(rand.best_score_)
print(rand.best_params_)

#%time predicted = model.predict(testWrangled)
#dfWithClass = pd.DataFrame(predicted, columns = ['Class'])
#final = pd.concat([testWrangled, dfWithClass], axis=1)
#convert the enumerated class labels to descriptive labels
#classLabels = unIndex(categoryLabelsTraining, final['Class'])
#categoriesPredicted = pd.get_dummies(classLabels)

#if there are any missing labels, append them to the end so that the output is complete as required by kaggle
#missingClasses = findMissingLabels(categoryLabelsTraining, classLabels)
#empty = pd.DataFrame(0, index = np.arange(len(classLabels)), columns = missingClasses)

#finalFormatted = pd.concat([categoriesPredicted, empty], axis=1)
#finalFormatted.to_csv("../data/predictions.csv")


# In[ ]:




# In[ ]:



