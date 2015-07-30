import pandas as pd
import patsy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

startTime = time.time()

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


train = pd.read_csv("./train.csv",
                    usecols = ['Category','DayOfWeek','PdDistrict','X','Y']
                    )
test = pd.read_csv("./test.csv",
                   usecols = ['DayOfWeek','PdDistrict','X','Y']
                   )

def unIndex(labels, enumIndex):
    return [labels[index] for index in enumIndex]

trainingWrangled, enumIndexTraining, categoryLabelsTraining = wrangle(train, True)
testWrangled, enumIndexTest, categoryLabelsTest = wrangle(test, False)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trainingWrangled, enumIndexTraining)
predicted = rf.predict(testWrangled)
dfWithClass = pd.DataFrame(predicted, columns = ['Class'])
final = pd.concat([testWrangled, dfWithClass], axis=1)
classLabels = unIndex(categoryLabelsTraining, final['Class'])
categoriesPredicted = pd.get_dummies(classLabels)
categoriesPredicted.to_csv("./predictions.csv")
print("Time taken:%.1f" % (time.time() - startTime))






