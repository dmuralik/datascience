
# coding: utf-8

# In[109]:

import requests, json
import pandas as pd
import xml.etree.ElementTree as ET

#extract the sources
sourcesUrl = 'http://api.worldbank.org/sources?format=json'
response = requests.post(sourcesUrl)
sources = pd.DataFrame(response.json()[1])
sources.drop(['description', 'url'], inplace = True, axis = 1)

#get data for gender statistics
genderStatId = sources[sources.name == 'Gender Statistics'].id.values[0]
genderStatUrl = 'http://api.worldbank.org/source/' + genderStatId + '/indicators'  
response = requests.post(genderStatUrl)
root = ET.fromstring(response.content)

elementPrefix = '{http://www.worldbank.org}'
indicatorId = []
indicatorNames = []
for indicator in root.iter(elementPrefix+'indicator'):
    indicatorId.append(indicator.attrib['id'])
for name in root.iter(elementPrefix+'name'):
    indicatorNames.append(name.text)
genderIndicators = pd.DataFrame(indicatorId, columns = ['Id'])
genderIndicators['Name'] = indicatorNames
print(genderIndicators['Name'])

countriesUrl = 'http://api.worldbank.org/countries?per_page=300&format=json'
response = requests.post(countriesUrl)
countries = pd.DataFrame(response.json()[1])

#264 (countries) *54(1960 until 2014)*50(indicators) = 712800 data points just for gender statistics - 45mb
#WDI - 45mb
#EdStats - 165mb
#World Development Report 2013 on Jobs Statistical Tables - 1 mb
#Global Financial Inclusion (Global Findex) Database - 17.5 mb

#indicatorUrl = 'http://api.worldbank.org/countries/all/indicators/'
#for row in genderIndicators.itertuples():
    #response = requests.post(indicatorUrl + row[1] + '?format=json')
    #print(response.json())
    #break


# In[ ]:



