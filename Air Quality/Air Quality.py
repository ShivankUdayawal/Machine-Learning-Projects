#!/usr/bin/env python
# coding: utf-8

# # Air Quality
# 
# 
# ## Introduction :
# 
# Air pollution stands as a pressing global concern with far-reaching implications for both human health and environmental well-being. Prolonged exposure to suboptimal air quality, stemming from diverse sources such as vehicular emissions, industrial effluents, mold spores, and wildfires, manifests in a spectrum of adverse health effects. These include irritation of the eyes, nose, and throat, as well as respiratory challenges, cardiovascular complications, and, over extended durations, more serious health ramifications.
# 
# Beyond the human toll, the environmental repercussions of compromised air quality are equally severe. Impaired air quality has been linked to diminished crop yields, heightened vulnerability of plants to pests and diseases, and broader ecological imbalances.
# 
# Vigilant monitoring of air quality levels assumes paramount importance for safeguarding both human health and the environment. To this end, this project embarks on an exploration of the AirQuality Dataset. The objective is to develop a robust machine learning (ML) model capable of predicting temperature based on concentrations of various pollutants, including metal oxides and hydrocarbons. This undertaking not only addresses the imperative of air quality management but also provides an opportunity to delve into the intricacies of working with time-series data.
# 
# 
# ### Dataset : [Air Quality](!https://archive.ics.uci.edu/dataset/360/air+quality)
# 
# 
# ## Dataset Information
# 
# The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2)  and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.
# This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.
# 
# ### Attribute Information
# 
# * Date (DD/MM/YYYY)
# 
# * Time (HH.MM.SS)
# 
# * True hourly averaged concentration CO in mg/m^3 (reference analyzer)
# 
# * PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
# 
# * True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
# 
# * True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
# 
# * PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
# 
# * True hourly averaged NOx concentration in ppb (reference analyzer)
# 
# * PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
# 
# * True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
# 
# * PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
# 
# * PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
# 
# * Temperature in Â°C
# 
# * Relative Humidity (%)
# 
# * AH Absolute Humidity

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from tqdm import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import seaborn as sns

import datetime

from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel , RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import pickle
import scipy.stats as stats

# For warnings
import warnings
warnings.filterwarnings("ignore")


# ### Importing the Dataset

# In[2]:


df = pd.read_csv("AirQuality.csv", sep = ";", decimal = ",")        
df.head()


# In[3]:


df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True, errors = 'ignore') 
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe().T


# * **Missing values are tagged with -200 value.**

# In[7]:


# Replacing bad sensor readings designated by an entry of -200 with NaN
df.replace(to_replace = -200, value = np.nan, inplace = True)


# In[8]:


df.info()


# In[9]:


df.describe().T


# In[10]:


df.isnull().sum()


# In[11]:


round( 100*( df.isnull().sum() / len(df.index) ), 2 ).sort_values(ascending = False)


# ### Removing Null Values

# In[12]:


df.drop('NMHC(GT)', axis = 1, inplace = True, errors = 'ignore') 

df.head()


# In[13]:


df[df['Date'].isnull()]


# In[14]:


df = df.dropna()


# In[15]:


df.shape


# In[16]:


df.info()


# ### Exploratory Data Analysis

# In[17]:


df['DateTime'] =  df['Date'] + ' ' + df['Time']

df.DateTime = df.DateTime.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H.%M.%S'))
df.head()


# In[18]:


df['Weekday'] = df['DateTime'].dt.day_name()
df['Month']   = df['DateTime'].dt.month_name()
df['Hour']    = df['DateTime'].dt.hour

df['Date']    = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

df.drop('Time', axis = 1, inplace = True, errors = 'ignore') 

df = df[['Date', 'Month', 'Weekday', 'Hour', 'DateTime', 'CO(GT)','PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 
         'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]

df.head()


# In[19]:


for i in df.columns:
    print("Count of unique values in \033[1m{}\033[0m column are \033[1m{}\033[0m.".format(i, df[i].nunique()))


# In[20]:


for i in df.columns:
    print("Minimum and Maximum value in \033[1m{}\033[0m column are \033[1m{}\033[0m and \033[1m{}\033[0m respectively.".format(i, df[i].min(), df[i].max()))


# In[21]:


df.info()


# In[22]:


df.columns


# In[23]:


l = ['Date', 'Hour', 'DateTime', 'CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 
     'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
for i in l:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3-q1
    upper = q3 +1.5*iqr
    lower = q1 -1.5*iqr
    outliner_df = df.loc[(df[i] < lower)|(df[i] > upper)]
    print('Percentage of outlier in \033[1m{0}\033[0m column is \033[1m{1}\033[0m%.'.format(i, round((outliner_df.shape[0]/df.shape[0])*100),2))


# In[24]:


l = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
     'T', 'RH', 'AH']

plt.figure(figsize = (20,30))
j = 1
for i in l:
    plt.subplot(6, 2, j)
    sns.boxplot(df[i])
    plt.title("Boxplot of {}".format(i))
    plt.xticks(rotation = 30)
    j = j+1
plt.tight_layout()


# ### Removing Outliers

# In[25]:


# Removing Outliers with the Interquartile Range Method (IQR)
Q1 = df.quantile(0.25) #first 25% of the data
Q3 = df.quantile(0.75) #first 75% of the data
IQR = Q3 - Q1 #IQR = InterQuartile Range

scale = 1.4 # May need to play with this value to modify outlier detection sensitivity if need be
lower_lim = Q1 - scale*IQR
upper_lim = Q3 + scale*IQR

cols = df.columns[5:] # Look for oulierts in columns starting from CO(GT)

# Mask a masking condition that removes rows that have values above/below IQR limits
condition = ~((df[cols] < lower_lim) | (df[cols] > upper_lim)).any(axis = 1)

# Generate new dataframe that has had its outliers removed
df_filtered = df[condition]
df_filtered.head()


# In[26]:


df_filtered.info()


# In[27]:


df.reset_index(drop = True, inplace = True)
report = ProfileReport(df)
report


# In[28]:


df_filtered.reset_index(drop = True, inplace = True) 
report = ProfileReport(df_filtered)
report


# * **C6H6** is a hydrocarbon and we have another sensor that is detecting nonmetallic hydrocarbons (NMHC). Therefore, the C6H6 column is sorta redundant and we can drop it without losing too much information.
# 
# * **NOx**, we still have a sensor in the dataframe responsible for detecting nitrous oxides. This means that the NOx column is similarly redundant.
# 
# * **CO(GT)**, we still have a sensor in the dataframe responsible for specifically detecting CO. This means that the CO(GT) column is also redundant.
# 
# * Using that same logic, I'll drop the **NO2(GT)** column as well.

# In[29]:


df_filtered.drop(['CO(GT)']  , axis = 1, inplace = True, errors = 'ignore')
df_filtered.drop(['NOx(GT)'] , axis = 1, inplace = True, errors = 'ignore')
df_filtered.drop(['C6H6(GT)'], axis = 1, inplace = True, errors = 'ignore')
df_filtered.drop(['NO2(GT)'] , axis = 1, inplace = True, errors = 'ignore')
df_filtered.head()


# In[30]:


df_filtered.reset_index(drop = True, inplace = True) 
report = ProfileReport(df_filtered)
report


# ### Time Series on Air Quality Data

# In[31]:


month_df_list = []
day_df_list   = []
hour_df_list  = []

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 
          'December']

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for month in months:
    temp_df = df_filtered.loc[(df_filtered['Month'] == month)]
    month_df_list.append(temp_df)

for day in days:
    temp_df = df_filtered.loc[(df_filtered['Weekday'] == day)]
    day_df_list.append(temp_df)

for hour in range(24):
    temp_df = df_filtered.loc[(df_filtered['Hour'] == hour)]
    hour_df_list.append(temp_df)


# In[32]:


def df_time_plotter(df_list, time_unit, y_col):
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 
              'December']
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if time_unit == 'M':
        nRows = 3
        nCols = 4
        n_iter = len(months)
    elif time_unit == 'D':
        nRows = 2
        nCols = 4
        n_iter = len(days)
    elif time_unit == 'H':
        nRows = 4
        nCols = 6
        n_iter = 24
    else:
        print('time_unit must be a string equal to M,D, or H')
        return 0
        
    fig, axs = plt.subplots(nrows = nRows, ncols = nCols, figsize = (40, 30))
    axs = axs.ravel()
    for i in range(n_iter):
        data = df_list[i]
        ax = axs[i]
        data.plot(kind ='scatter', x = 'DateTime', y = y_col , ax = ax, fontsize = 24)
        ax.set_ylabel('Pollutant Concentration', fontsize = 30)
        ax.set_xlabel('')
        if time_unit == 'M':
            ax.set_title(y_col + ' ' + months[i],  size = 40) 
        elif time_unit == 'D':
            ax.set_title(y_col + ' ' + days[i],  size = 40) 
        else:
             ax.set_title(y_col + ' ' + str(i),  size = 40) 
        ax.tick_params(labelrotation = 60)

        
    # set the spacing between subplots
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.4,  hspace = 0.5)
    plt.show() 


# In[33]:


df_time_plotter(month_df_list, 'M', 'PT08.S3(NOx)')


# In[34]:


df_time_plotter(day_df_list, 'D', 'PT08.S3(NOx)')


# In[35]:


df_time_plotter(hour_df_list, 'H', 'PT08.S3(NOx)')


# In[36]:


plt.figure(figsize = (18, 6))

sns.barplot(x = 'Month', y = 'PT08.S3(NOx)', data = df_filtered)
plt.title('NOx Values Per Month')
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize = (18, 6))
sns.barplot(x = 'Weekday', y = 'PT08.S3(NOx)', data = df_filtered)
plt.title('NOx Values Per Day of the Week')
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize = (18, 6))
sns.barplot(x = 'Hour', y = 'PT08.S3(NOx)', data = df_filtered)
plt.title('NOx Values Per Hour')
plt.xticks(rotation = 90)
plt.show()


# In[37]:


df_final = df_filtered.iloc[:, 5:]
df_final


# In[38]:


sns.set(font_scale = 1.5)
sns.pairplot(df_final)


# In[39]:


df_final.info()


# ### PreProcessing Data

# In[40]:


X = df_final.drop(columns = ['PT08.S3(NOx)'])
y = df_final['PT08.S3(NOx)']


# In[41]:


print(X.shape)
print(y.shape)


# ### Splitting the dataset and Building Machine Learning Models

# In[42]:


def model_assess(X_train, X_test, y_train, y_test, model, title = "Default"):
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2  = r2_score(y_train, y_train_pred)
    test_mse  = mean_squared_error(y_test, y_test_pred)
    test_r2   = r2_score(y_test, y_test_pred)
    
    results = pd.DataFrame([title,train_mse, train_r2, test_mse, test_r2]).transpose()
    results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']
    return y_train_pred, y_test_pred, results

def multi_model_assess(df, models, y_predict):
    all_model_results = [] #This will contain all model results for each dependent variable 
    all_X_test    = []
    all_X_train   = []
    all_y_test_p  = []
    all_y_train_p = []
    all_y_train   = []
    
    #First loop will define dependent/independent variables and split data into test/training sets
    n_vars = len(y_predict)
    pbar = tqdm(range(n_vars), desc = "Variable Processed", position = 0, leave = True) #Add progress bar 
    
    for dependent in y_predict:
        model_results = [] #Array with dataframes for a given dependent variable
        
        #Designate independent and dependent variables
        x  = df.drop([dependent], axis = 1)
        y  = df[dependent]
        
        #Split data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
        
        #Populate the array of observed values for the dependent variable
        all_y_train.append(y_train)
        
        #Process each of the desired models
        for model, model_name in models:
            y_train_pred,y_test_pred, results = model_assess(X_train, X_test, y_train, y_test, model, title = model_name)
            
            model_results.append(results)
            all_X_test.append(X_test)
            all_X_train.append(X_train)
            all_y_test_p.append(y_test_pred)
            all_y_train_p.append(y_train_pred)
                    
        all_model_results.append(model_results)
        pbar.update(1)
        pbar.refresh()
        
    pbar.close()   
    return all_model_results, all_X_test, all_X_train, all_y_test_p, all_y_train_p, all_y_train


# In[43]:


#Initiate Different Regressors for ML model
lr = LinearRegression() 
rf = RandomForestRegressor(n_estimators = 100, max_depth = 3, random_state = 42)
gb = GradientBoostingRegressor(n_estimators = 100, max_depth = 3, random_state = 42)
kn = KNeighborsRegressor()
ab = AdaBoostRegressor()
sv = SVR()
nn = MLPRegressor(hidden_layer_sizes = 500, solver = 'adam', learning_rate_init = 1e-2, max_iter = 500)

models =  [(lr,'Linear Regression'), 
           (rf,'Random Forest'), 
           (gb,'Gradient Boosting'),
           (kn,'K-Neighbors'), 
           (ab,'Ada Boost'), 
           (sv,'SVR'), 
           (nn,'MLP')]

y_predict  = ['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']

all_model_results, _, _, all_y_test_p, all_y_train_p, all_y_train = multi_model_assess(df_final, models, y_predict)


# In[44]:


score_df_results = pd.concat(all_model_results[0], ignore_index = True).sort_values('Test R2', axis = 0, ascending = False)
score_df_results


# In[45]:


score_results_test = pd.concat(all_model_results[0], ignore_index = True)
score_results_test['Test R2'][0]


# In[46]:


#Define column with model to predict
y_predict  = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']

#Model names for plot titles
models =  [(lr,'Linear Regression'), 
           (rf,'Random Forest'), 
           (gb,'Gradient Boosting'),
           (kn,'K Neighbors'), 
           (ab,'Ada Boost'), 
           (sv,'SVR'), 
           (nn,'MLP')]

#Make labels
def make_labels(models):
    names = []
    for i in range(len(models)):
        if len(models[i][1].split()) < 2:
            names.append(models[i][1])
        else:
            names.append(''.join([s[0] for s in models[i][1].split()]))
    return names
labelList = make_labels(models)

#Specify color map to color different plots
cmap = plt.cm.get_cmap('plasma')
slicedCM = cmap(np.linspace(0, 1, len(models))) 

#Visualize results of linear regression
plt.rcParams.update({'font.size': 20})
nRows = 4  
nCols = 2 

def plot_ML_model(whichVar):
    fig, axs = plt.subplots(nrows = nRows, ncols = nCols, figsize = (15, 30))
    axs = axs.ravel()
    df    = pd.concat(all_model_results[whichVar], ignore_index = True)
    for k in range(7):
        color = slicedCM[k]
        yPred = all_y_train_p[k + whichVar*len(models)]
        yMeas = all_y_train[whichVar]
        label = labelList[k]
        ax    = axs[k]
        
        #Make scatter plot of train set and regressor model
        ax.scatter(x = yMeas, y = yPred, color = color, alpha = 0.5)

        #Fit a first order polynomial (i.e. a straight line) to the regressor model 
        z = np.polyfit(yMeas, yPred, 1)
        p = np.poly1d(z)

        #Add labels and colors and stuff
        val = df['Test R2'][k] #Get the r2 value from the model results dataframe
        val = "{:.2f}".format(val)
        ax.plot(yMeas, p(yMeas), "#b20cd7", label = label +"\nr\u00b2".format(2) + " = " + str(val))
        ax.title.set_text(models[k][1]) 
        ax.set(xlabel = 'Train Concentration', ylabel = 'Predicted Concentration')
        ax.label_outer()
        ax.legend(loc = "upper left")
        ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.show()


# In[47]:


plot_ML_model(0)


# In[48]:


score_df_results = pd.concat(all_model_results[1], ignore_index = True).sort_values('Test R2', axis = 0, ascending = False)
score_df_results


# In[49]:


plot_ML_model(1)


# In[50]:


score_df_results = pd.concat(all_model_results[2], ignore_index = True).sort_values('Test R2', axis = 0, ascending = False)
score_df_results


# In[51]:


plot_ML_model(2)


# In[52]:


score_df_results = pd.concat(all_model_results[3], ignore_index = True).sort_values('Test R2', axis = 0, ascending = False)
score_df_results


# In[53]:


plot_ML_model(3)


# In[54]:


score_df_results = pd.concat(all_model_results[4], ignore_index = True).sort_values('Test R2', axis = 0, ascending = False)
score_df_results


# In[55]:


plot_ML_model(4)


# ### Deep Learning Model - LSTM

# In[56]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[57]:


from sklearn.preprocessing import MinMaxScaler
x_train = x_train.values
x_test  = x_test.values
y_train = y_train.values
y_test  = y_test.values


# In[58]:


from tensorflow.keras.callbacks import EarlyStopping

# Reshape the data to a 3D tensor [Batchsize, timstamps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test  = x_test.reshape(x_test.shape[0],   1, x_test.shape[1])


# In[59]:


print(x_train.shape)
print(x_test.shape)


# In[60]:


import tensorflow as tf

# Define your LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.LSTM(128, return_sequences = True,), 
    tf.keras.layers.LSTM(128, return_sequences = True),
    tf.keras.layers.LSTM(128, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


# In[61]:


# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[62]:


# Train the model
model.fit(x_train, y_train, epochs = 250, batch_size = 32)


# In[63]:


# Make predictions
y_pred = model.predict(x_test)


# In[64]:


y_pred.shape


# In[65]:


y_test.shape


# In[66]:


# Flatten the predictions and ground truth labels
y_pred_flat = y_pred.flatten()


# In[67]:


# Evaluate your model using Scikit-learn
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred_flat)
print(f'Mean Squared Error: {mse}')


# In[ ]:





# In[ ]:




