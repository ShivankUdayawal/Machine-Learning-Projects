#!/usr/bin/env python
# coding: utf-8

# # Air Quality
# 
# 
# ![image.jpg](attachment:image.jpg)
# 
# ### Introduction:
# 
# India grapples with severe air pollution issues, prompting a robust governmental initiative to enhance air quality monitoring. This study presents the inaugural comprehensive analysis of government-generated air quality observations spanning 2015 to 2020. Leveraging data from the Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring (CAAQM) network, the manual National Air Quality Monitoring Program (NAMP), and the US Air-Now network for PM2.5, our research meticulously addresses data inconsistencies and gaps to ensure representativeness.
# 
# Particular attention is given to PM10, PM2.5, SO2, NO2, and O3 pollutants, revealing that particulate pollution predominantly dominates the pollution spectrum nationwide. Notably, in northern India (divided at 23.5°N), virtually all monitoring sites surpass the annual average PM10 and PM2.5 residential National Ambient Air Quality Standards (NAAQS) by 150% and 100%, respectively. Southern India, while showing relatively lower concentrations, exceeds the PM10 standard by 50% and the PM2.5 standard by 40%. Conversely, annual average SO2, NO2, and MDA8 O3 levels generally comply with residential NAAQS standards across the country.
# 
# Regional differentials are pronounced, with northern India exhibiting concentrations 10% to 130% higher than southern counterparts, with SO2 being the exception. Despite discernible inter-annual variability, our study does not identify significant trends in these pollutants over the five-year period. In the five cities equipped with Air-Now PM2.5 measurements – Delhi, Kolkata, Mumbai, Hyderabad, and Chennai – a commendable alignment is observed with CPCB data. Furthermore, the PM2.5 CPCB CAAQM data aligns favorably with satellite-derived annual surface PM2.5 concentrations, except for the western desert region, where surface measurements exceeded satellite retrievals prior to 2018.
# 
# This reanalyzed dataset offers valuable insights for evaluating Indian air quality through satellite data, atmospheric models, and low-cost sensors. Additionally, it establishes a baseline for assessing the efficacy of the National Clean Air Programme and aids in evaluating current and prospective air pollution mitigation policies.
# 
# 
# ### Content:
# 
# The dataset contains air quality data and AQI (Air Quality Index) at hourly and daily level of various stations across multiple cities in India.
# 
# ### Dataset:
# 
# The data has been made publicly available by the Central Pollution Control Board: [Link](https://cpcb.nic.in/) which is the official portal of Government of India. 
# 
# The dataset is a collection of pollutant readings across cities in India recorded between 2015 and 2020. The data consists of 26 cities in India, and is split into the following categories: -
# 
#      Date - daily readings between 2015 and 2020
# 
#      PM2.5 - Particulate Matter 2.5-micrometer in ug / m3
# 
#      PM10 - Particulate Matter 10-micrometer in ug / m3
# 
#      NO - Nitric Oxide in ug / m3
# 
#      NO2 - Nitric Dioxide in ug / m3
# 
#      NOx - Any Nitric x-oxide in ppb
# 
#      NH3 - Ammonia in ug / m3
# 
#      CO - Carbon Monoxide in mg / m3
# 
#      SO2 - Sulphur Dioxide in ug / m3
# 
#      O3 - Ozone in ug / m3
# 
#      Benzene - Benzene in ug / m3
# 
#      Toluene - Toluene in ug / m3
# 
#      Xylene - Xylene in ug / m3
# 
#      AQI - Air Quality Index
# 
#      AQI Bucket - Air Quality Index Bucket (ranging from 'Good' to 'Hazardous')
# 
# 
# ### Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import os, sys 
import sqlite3
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objs as go

# For warnings
import warnings
warnings.filterwarnings("ignore")


# ### Importing the Dataset

# In[2]:


df = pd.read_csv("city_day.csv")
df.head()


# In[3]:


df.shape


# ### Summary of Dataset

# In[4]:


df.info()


# * **So many missing values**

# In[5]:


df.describe().T


# In[6]:


df.isnull().sum()


# In[7]:


missing_values = df.isnull().sum()
percentage_missing = (missing_values / len(df) * 100).round(2)

missing_values_table = pd.DataFrame({'Column Name': missing_values.index, 
                                     'No. of Missing Values': missing_values.values, 
                                     '% of Missing Values': percentage_missing.values})

missing_values_table


# ### Exploratory Data Analysis
# 
# #### Number of Readings per City

# In[8]:


city = df['City'].value_counts().to_frame().reset_index().rename(columns = {'index':'City Name', 
                                                                                     'City':'No. of readings'})
city['%'] = (100* city['No. of readings']/city['No. of readings'].sum()).round(0)
city


# #### Number of Readings per Year 

# In[9]:


df['Date'] = pd.to_datetime(df['Date'])
annual_readings = df.Date.dt.year.value_counts().to_frame().reset_index().rename(columns = {'index':'Year', 
                                                                                            'Date':'No. of readings'})
annual_readings['%'] = (100* annual_readings['No. of readings']/annual_readings['No. of readings'].sum()).round(0)
annual_readings.sort_values(by = ['Year'])


# * Here, as we see in year 2015 only 2801 no of reading which is only 9% which is lowest and for year 2019 no of reading is 7446 which is 25% and highest.
# 
# * we see an increasing trend from 2015 to 2019 that is maximum and then a dip again in year 2020.
# 
# #### Data Cleaning
# 
# * Replace the missing values for numerical columns with mean

# In[10]:


df['PM2.5'].fillna(df['PM2.5'].mean().round(2), inplace = True )
df['PM10'].fillna(df['PM10'].mean().round(2), inplace = True )
df['NO'].fillna(df['NO'].mean().round(2), inplace = True )
df['NO2'].fillna(df['NO2'].mean().round(2), inplace = True )
df['NOx'].fillna(df['NOx'].mean().round(2), inplace = True )
df['NH3'].fillna(df['NH3'].mean().round(2), inplace = True )
df['CO'].fillna(df['CO'].mean().round(2), inplace = True )
df['SO2'].fillna(df['SO2'].mean().round(2), inplace = True )
df['O3'].fillna(df['O3'].mean().round(2), inplace = True )
df['Benzene'].fillna(df['Benzene'].mean().round(2), inplace = True )
df['Toluene'].fillna(df['Toluene'].mean().round(2), inplace = True )
df['Xylene'].fillna(df['Xylene'].mean().round(2), inplace = True )
df['AQI'].fillna(df['AQI'].mean().round(2), inplace = True )


# In[11]:


df.head()


# In[12]:


df['AQI_Bucket'].isnull().sum()


# In total, there are 4681 missing descriptions in the 'AQI Bucket' column. Let's populate the missing rows in accordance with the score system implemented.
# 
# 
# ![38761.jpg](attachment:38761.jpg)
# 
#  AQI Score System: -
# 
#     * Good (0-50)
#     * Moderate (51-100)
#     * Unhealthy for Sensitive groups (101-150)
#     * Unhealthy (151-200)
#     * Very Unhealthy (201-300)
#     * Hazardous (301-500)

# In[13]:


df.loc[(df['AQI'] >= 0)   & (df['AQI'] <= 50),  'AQI_Bucket'] = 'Good'
df.loc[(df['AQI'] >= 51)  & (df['AQI'] <= 100), 'AQI_Bucket'] = 'Moderate'
df.loc[(df['AQI'] >= 101) & (df['AQI'] <= 150), 'AQI_Bucket'] = 'Unhealthy for Sensitive groups'
df.loc[(df['AQI'] >= 151) & (df['AQI'] <= 200), 'AQI_Bucket'] = 'Unhealthy'
df.loc[(df['AQI'] >= 201) & (df['AQI'] <= 300), 'AQI_Bucket'] = 'Very Unhealthy'
df.loc[(df['AQI'] >= 301) & (df['AQI'] <= 500), 'AQI_Bucket'] = 'Hazardous'


# In[14]:


df['AQI_Bucket'].isnull().sum()


# #### Average Readings by City 

# In[15]:


avg_readings = df.groupby('City').mean().reset_index().round(2)
avg_readings


# In[16]:


avg_readings.describe().T


# ![image1.jpg](attachment:image1.jpg)

# #### Average Readings for Cities against each pollutant

# In[17]:


for column in avg_readings.columns[1:]:
    fig = go.Figure(data = [go.Pie(labels = avg_readings['City'], values = avg_readings[column].round(2), 
                                   title = f'Average {column} by City')])

    fig.update_traces(textposition = 'inside', textinfo = 'percent+label')
    fig.update_layout(uniformtext_minsize = 12, uniformtext_mode = 'hide', title = f'Average {column} by City')
    fig.show()


# #### On average, which city scores highest and lowest for each pollutant?

# In[18]:


avg_readings.round(2).head()


# In[19]:


avg_readings.head()


# In[20]:


MaxVals = avg_readings[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3','Benzene', 'Toluene', 'Xylene', 
                        'AQI']].dropna().idxmax()
MaxVals


# In[21]:


MaxCities = avg_readings.iloc[[10, 10, 17, 10, 17, 8, 0, 0, 5, 22, 22, 3, 0], [0]].reset_index().rename(
    columns = {'City':'City, Max. Readings'})
MaxCities


# In[22]:


MinVals = avg_readings[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 
                        'AQI']].dropna().idxmin()
MinVals


# In[23]:


MinCities = avg_readings.iloc[[20, 19, 22, 1, 16, 0, 22, 11, 11, 5, 5, 1, 1], [0]].reset_index().rename(
    columns = {'City':'City, Min. Readings'})
MinCities


# In[24]:


pollutants = avg_readings.columns.to_frame().drop(labels = 'City', axis = 0).reset_index()
pollutants


# In[25]:


summ_max_min = pd.concat([pollutants, MaxCities, MinCities], axis = 1).drop(columns = ['index'])
summ_max_min.rename(columns = {0:'Pollutant'} )


# * Here, it is clear that:
# 
# * **Delhi scores highest in the readings for PM2.5, PM10 and NO2.** 
# 
# * **Kochi also scores highest against NO and NOx which is concerning given the low quantity of readings.**
# 
# * **Exception of Mumbai and Lucknow, the cities with the lowest scores are Shillong, Ernakulam and Aizwal which are also the cities with the lowest quantity of readings between 2015 and 2020.**
# 
# 
# 
# 
# 
# 
# #### Pollutants Carbon Monoxide (CO) and Sulfur Dioxide (SO2)
# 
# #### Avg CO readings in order of high to low by City 

# In[26]:


avg_co = avg_readings[['City', 'CO']].sort_values('CO', ascending = False)
avg_co


# #### Avg SO2 readings in order of high to low by City 

# In[27]:


avg_so2 = avg_readings[['City', 'SO2']].sort_values('SO2', ascending = False)
avg_so2


# In[28]:


city_annual = df.copy()

city_annual['Date'] = pd.to_datetime(city_annual['Date'])
city_annual = city_annual.groupby(['City', city_annual.Date.dt.year]).mean().reset_index().rename(columns = {'Date':'Year'})
city_annual.head()


# In[29]:


fig = px.bar(city_annual, x = 'Year', y = 'CO', color = 'City', barmode = 'group', 
             title = 'Carbon Monoxide (CO) average readings for each city between 2015 - 2020')
fig.show()


# In[30]:


fig = px.bar(city_annual, x = 'City', y = 'CO' , color = 'Year', 
             title = 'Annual Average Carbon Monoxide (CO) Readings for each City, 2015-2020')
fig.show()


# In[31]:


fig = px.bar(city_annual, x = 'Year', y = 'SO2', color = 'City', barmode = 'group', 
             title = 'Sulfur Dioxide (SO2) average readings for each city between 2015 - 2020')
fig.show()


# In[32]:


fig = px.bar(city_annual, x = 'City', y = 'SO2' , color = 'Year', 
             title = 'Annual Average Sulfur Dioxide (SO2) Readings for each City, 2015-2020')
fig.show()


# * **Ahmedabad scores highly on CO and SO2 between 2015 and 2020**, which suggests that the city needs to consider enforcing some policies to reduce air pollution. 
# 
# * In some cases such as ***Delhi and Lucknow*** it is evident that the local government has implemented some policies to improve the air quality as the readings have decreased over the 5-year period.
# 
# 
# #### Pollutants PM2.5 and NO2

# In[33]:


trace5 = go.Box(x = city_annual['PM2.5'], name = 'PM2.5')
d5 = [trace5]
layout = go.Layout(title = 'Range of PM2.5 readings')
fig = go.Figure(data = d5, layout = layout)
fig.show()


# In[34]:


fig = px.bar(city_annual, x = 'Year', y = 'PM2.5', color = 'City', barmode = 'group', 
             title = 'PM2.5 average readings for each city between 2015 - 2020')
fig.show()


# In[35]:


fig = px.bar(city_annual, x = 'City', y = 'PM2.5' , color = 'Year', 
             title = 'Annual Average PM2.5 Readings for each City, 2015-2020')
fig.show()


# * Here, we can see for **PM2.5** has expanded over the years to include more cities. 
# 
# * ***Delhi, Lucknow and Patna have maintained a steady level of PM2.5.***
# 
# * However, it is fair to say that most cities had a lower reading in 2020 - this could be due to the Pandemic when more people were working from home.

# In[36]:


fig = px.bar(city_annual, x = 'Year', y = 'NO2', color = 'City', barmode = 'group', 
             title = 'NO2 average readings for each city between 2015 - 2020')
fig.show()


# In[37]:


trace4 = go.Box(x = city_annual['NO2'], name = 'NO2')
d4 = [trace4]
layout = go.Layout(title = 'Range of NO2 readings')
fig = go.Figure(data = d4, layout = layout)
fig.show()


# In[38]:


fig = px.bar(city_annual, x = 'City', y = 'NO2' , color = 'Year', title = 'Annual Average NO2 Readings for each City')
fig.show()


# Similar to the PM2.5 readings, the NO2 dataset has also expanded over time to include more cities. Ahmedabad appears to have high readings for 2018 and 2019, whereas the readings for Delhi are steadily decreasing over the years. This could be due to measures implemented to try to reduce the pollution in the city.* Herer
# 
# 
# 
# 
# ### Air Quality Index (AQI)

# In[39]:


aqi_avg_df = city_annual.copy().drop(columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 
                                                'Toluene', 'Xylene'])
aqi_avg_df


# In[40]:


aqi_avg_df['Average_AQI'] = aqi_avg_df['AQI'].mean()
aqi_avg_df


# In[41]:


fig = px.bar(aqi_avg_df, x = 'Year', y = 'AQI', color = 'City', barmode = 'group', 
             title = 'AQI for each city between 2015 - 2020')
l = px.line(aqi_avg_df, x = 'Year', y = 'Average_AQI').update_traces(line_color = "black", name = 'Average AQI', 
                                                                     showlegend = True)
fig.add_traces(l.data)
fig.show()


# * Most cities are scoring below average which is considered to be **"Unhealthy".** 
# 
# * The lower the AQI score, the better the rating. 
# 
# * It is clear that ***Ahemedabad has consistently performed badly in the AQI ratings.***
# 
# * The large drop between 2019 and 2020 is most likely due to the pandemic when more people were working from home amongst other factors. By increasing awareness of the problem we can help to drive practical solutions.

# In[42]:


aqi_bucket_df = df[['City', 'Date', 'AQI_Bucket']] 
aqi_bucket_df


# In[43]:


aqi_bucket_df_by_yr = aqi_bucket_df.groupby(['City', aqi_bucket_df.Date.dt.year, 'AQI_Bucket']).count().rename(
    columns = {'Date':'Quantity'}).reset_index()
aqi_bucket_df_by_yr = aqi_bucket_df_by_yr.rename(columns = {'Date':'Year'})
aqi_bucket_df_by_yr


# In[44]:


fig = px.bar(aqi_bucket_df_by_yr, x = 'City', y = 'Quantity' , color = 'AQI_Bucket', 
             title = 'Annual Air Quality Index by City, 2015-2020')
fig.show()


# * The 5 cities with the worst AQI performance are: - **Delhi, Lucknow, Patna, Gurugram and Ahmedabad.** 
# 
# * The 5 cities with the best AQI performance are: - **Amaravati, Hyderabad, Thiruvananthapuram, Kolkata and Guwahati.**
# 
# * What concepts could be adopted from the better performing cities such as Hyderabad and Chennai.

# In[45]:


fig = px.histogram(aqi_bucket_df_by_yr, x = 'AQI_Bucket', title = 'AQI Rating Histogram', )
fig.show()


# In[ ]:




