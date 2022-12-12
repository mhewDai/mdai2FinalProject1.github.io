#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stepwise_regression.stepwise_regression as sr
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px


# In[2]:


import chart_studio

username = 'mdai2'
api_key = 'rIMtGZ1ysapR5W0OlGkl'

chart_studio.tools.set_credentials_file(username=username,api_key=api_key)

import chart_studio.plotly as py
import chart_studio.tools as tls


# In[3]:


#read data as csv file
weather = pd.read_csv('Paris_weather_data_2017.csv', encoding='ISO-8859-1',na_values=['-'])
#repairing inconsistent time format
weather['Date']=pd.to_datetime(weather['Date'])
weather.info()


# In[4]:


weather.head()


# In[5]:


#Cleaning up Data by getting rid of variables with too many missing data points
weather = weather.drop(columns = ['Events','high Gust Wind (km/h)'])
weather.info()


# In[6]:


#Using interpolation to fill in any missing data points so every variable has a count of 365
weather = weather.interpolate(method='ffill',limit_direction='forward')
weather.info()


# In[7]:


#initiating correlation matrix
heatmap = weather.corr()
heatmap


# In[8]:


#Creating heatmap with color
fig = px.imshow(heatmap,
                labels=dict(x="", y="", color="Strength"),
                x=list(heatmap.index),
                y=list(heatmap.index),
               title="Heatmap of Variables")
fig.update_xaxes(side="top",
                )
#Sending map into plotly cloud
# py.plot(fig,filename='heatmap',auto_open=True)
fig


# In[9]:


electricity = pd.read_csv('Historique_consommation_JOUR_2017.csv')
#Making use that the df has the correct Date format inorder to merge the two dfs
electricity['Date']=pd.to_datetime(electricity['Date'])
electricity.head()


# In[10]:


#merging the two dfs based on date
merge = pd.merge(left=weather,right=electricity,how='inner',left_on='Date',right_on='Date')
merge.head()


# In[24]:


plot = px.scatter(x = merge['avg Sea Level Press. (hPa)'],
                  y = merge['Energie journalière (MWh)'],
                  color=merge['avg Dew Point (°C)'],
                  size=merge['avg Humidity (%)'],
                  title="Scatterplot of Energy Consumption vs Avg Temp")
plot.update_layout(xaxis_title="Avg Sea Level Press. (hPa)", 
                   yaxis_title="Daily Energy Consumption(MWh)",
                   title={'y':0.9,
                          'x':0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                  hovermode="x unified")
#Sending plot to cloud
# py.plot(plot,filename='scatterplot',auto_open=True)
plot


# In[ ]:


plot = px.scatter(x = merge['avg Temp. (°C)'],
                  y = merge['Energie journalière (MWh)'],
                  color=merge['avg Dew Point (°C)'],
                  size=merge['avg Humidity (%)'],
                  title="Scatterplot of Energy Consumption vs Avg Temp")
plot.update_layout(xaxis_title="Avg Temp(°C)", 
                   yaxis_title="Daily Energy Consumption(MWh)",
                   title={'y':0.9,
                          'x':0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                  hovermode="x unified")
#Sending plot to cloud
# py.plot(plot,filename='scatterplot',auto_open=True)
plot


# In[12]:


#scatterplot of energy consumption with respect to avg temp
scatterplot = plt.scatter(np.array(merge['avg Temp. (°C)']).astype(int),
            np.array(merge['Energie journalière (MWh)']).astype(int))
#Average Temperature on x-axis
plt.xlabel('Average Temperature(°C)', fontname='Times New Roman',fontsize=14)
#Energy Consumption on y-axis
plt.ylabel('Energy(MWh)', fontname='Times New Roman',fontsize=14)
plt.title('Scatterplot of Energy Consumption vs Avg Temp', fontname='Times New Roman',fontsize=16)


# In[13]:


parameters = np.polyfit(np.array(merge['avg Temp. (°C)']).astype(int), 
                        np.array(merge['Energie journalière (MWh)']).astype(int), 
                        2, 
                        rcond=None, 
                        full=False, 
                        w=None, 
                        cov=False)

x = np.array(merge['avg Temp. (°C)']).astype(int)
y = np.array(merge['Energie journalière (MWh)']).astype(int)
y2 = x ** 2 * parameters[0] + x * parameters[1] + parameters[2]
plt.scatter(x,y)
plt.plot(x,y2,color='r')
plt.xlabel('Average Temperature(°C)', fontname='Times New Roman',fontsize=14)
#Energy Consumption on y-axis
plt.ylabel('Energy(MWh)', fontname='Times New Roman',fontsize=14)
plt.title('Scatterplot of Energy Consumption vs Avg Temp', fontname='Times New Roman',fontsize=16)
min_x=-parameters[1]/(2*parameters[0])
min_y=min_x**2*parameters[0]+min_x*parameters[1]+parameters[2]
plt.plot(min_x,min_y,'o',ms=10,color='black')


# In[14]:


px.scatter(merge, x=merge["Date"], y=merge['Energie journalière (MWh)'], animation_frame=merge["Date"], animation_group=merge['avg Dew Point (°C)'],
           size=merge['avg Humidity (%)'], color=merge['avg Temp. (°C)'], hover_name=merge['avg Temp. (°C)'],
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

