#This is for a Kaggle datascience competition. More information about it can be found here: https://www.kaggle.com/c/sf-crime/data
#I have used a RandomForest classifier for prediction.
import pandas as pd
import patsy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from sklearn import cross_validation
import scipy as sp

#to track the performance
startTime = time.time()

#a function that can mungle the data (for reusability)
def wrangle(dataFrame, isTraining):
    dayOfWeekChanged = pd.get_dummies(dataFrame['DayOfWeek'])
    pdDistrictChanged = pd.get_dummies(dataFrame['PdDistrict'])
    enumIndex = categoryLabels = []
    if(isTraining):
        enumIndex,categoryLabels = pd.factorize(dataFrame['Category'])
        categoryChanged = pd.DataFrame(enumIndex, columns = ['Category'])
        dropped = dataFrame.drop(['DayOfWeek', 'PdDistrict', 'Category'], axis=1)
        transformed = pd.concat([dropped, dayOfWeekChanged, pdDistrictChanged, categoryChanged], axis=1)
    else:
        category = pd.DataFrame([], columns = ['Category'])
        dropped = dataFrame.drop(['DayOfWeek', 'PdDistrict'], axis=1)
        transformed = pd.concat([dropped, dayOfWeekChanged, pdDistrictChanged, category], axis=1)
        transformed['Category'].fillna(9999, inplace=True)
    return (transformed, enumIndex, categoryLabels)


train = pd.read_csv("./data/train.csv",
                    usecols = ['Category','DayOfWeek','PdDistrict','X','Y']
                    )
test = pd.read_csv("./data/test.csv",
                   usecols = ['DayOfWeek','PdDistrict','X','Y']
                   )

#unindexes class labels to descriptions
def unIndex(labels, enumIndex):
    return [labels[index] for index in enumIndex]

trainingWrangled, enumIndexTraining, categoryLabelsTraining = wrangle(train, True)
testWrangled, enumIndexTest, categoryLabelsTest = wrangle(test, False)

rf = RandomForestClassifier(n_estimators=100)
cv = cross_validation.KFold(len(train), n_folds=5, indices=False)
results = []
for traincv, testcv in cv:
    probas = rf.fit(trainingWrangled[traincv], enumIndexTraining[traincv]).predict_proba(trainingWrangled[testcv])
    results.append(metrics.log_loss(enumIndexTraining[testcv], [x[1] for x in probas]))

print(results)
#rf.fit(trainingWrangled, enumIndexTraining)
#predicted = rf.predict(testWrangled)
#dfWithClass = pd.DataFrame(predicted, columns = ['Class'])
#final = pd.concat([testWrangled, dfWithClass], axis=1)
#classLabels = unIndex(categoryLabelsTraining, final['Class'])
#categoriesPredicted = pd.get_dummies(classLabels)
#categoriesPredicted.to_csv("./data/predictions.csv")
print("Time taken:%.1f seconds" % (time.time() - startTime))






