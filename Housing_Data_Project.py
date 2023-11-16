#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# !pip install missingno
import missingno as msno
import matplotlib as mpl


mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['figure.figsize'] = [25, 10]


df = pd.read_csv('household_power_consumption.txt',sep = ';',
                parse_dates={'dt':['Date','Time']},
                infer_datetime_format=True,
                low_memory=False, na_values=['nan','?'],
                index_col='dt')

 
# Print the number of rows and columns in the data
print('Number of rows and columns:', df.shape)

# Display the first 5 rows of the data
df.head(5)
 
# Display the last 5 rows of the data
df.tail(5)
 
# Get the information about the dataframe
print("\nInformation about the dataframe:")
print(df.info())
 
# we find our target variable, y to be
# (global_active_power*1000/60 - sub_metering_1 - 
# sub_metering_2 - sub_metering_3)

df['total_energy_consumption'] = df['Global_active_power']*1000/60 - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']


pd.Series(df[df.isnull()].index.year).value_counts().plot(kind='bar')
 
plt.figure(figsize = (25,10))
msno.matrix(df);
plt.title('Missing Values Matrix', fontweight = 'bold')
plt.ylabel('Observations present/missing in data', fontweight = 'bold')
plt.xlabel('Variables in the data', fontweight = 'bold')
plt.show()

 
sns.heatmap(df.isna(), cmap='YlGnBu', cbar=False)
plt.title('Missing Values')
plt.xlabel('Columns')
plt.ylabel('Date')
plt.show()

df.describe()

df = df.resample('D').mean()
df = df.fillna(df.apply('median'))
df.sample(1)
# df = df.fillna('linear')

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is a pandas dataframe
# Define the number of rows and columns for the subplots
nrows = 2
ncols = 4

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows, ncols)

# Loop over the column names and the axes objects
for col, ax in zip(df.columns, axes.flat):
    # Plot a displot for each column on each subplot
    sns.displot(data=df, x=col, kind='hist', ax=axes)

# Adjust the spacing and show the plot
plt.tight_layout()
plt.show()
 
df.isnull().sum()
 
df.head(2)
# df['weekday'] = pd.Series(df.index.weekday).apply(lambda x: 'Weekend' if x > 4 else 'Weekday')
# df.drop('weekday_weekend', axis=1, inplace=True)
weekdays = {'Monday': 'Weekday', 'Tuesday': 'Weekday', 'Wednesday': 'Weekday', 'Thursday': 'Weekday', 'Friday': 'Weekday', 'Saturday': 'Weekend', 'Sunday': 'Weekend'}
df['weekday'] = df.index.day_name().map(weekdays)
df.head(2)

plt.figure(figsize =  (25, 50))
n = 5
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(5):
    plt.subplot(5,1,i+1)
    sns.boxplot(x = df.index.dayofyear, 
                    y = df[names[i]])
    plt.xticks([])
    plt.title('Daily Time Series Boxplots of {}'.format(names[i]))


plt.show()

df['weekday_weekend'] = pd.Series(df.index.weekday).apply(lambda x: 'Weekend' if x > 4 else 'Weekday')

plt.figure(figsize =  (60, 60))
n = 3
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(n):
    plt.subplot(n,1,i+1)
    sns.boxplot(x = df.index.dayofyear, 
                y = df[names[i]],
                hue = df['weekday'],
                palette = ['white', 'orange'])
    plt.xticks([])
    plt.title('Daily Time Series Boxplots of {} by Weekday/Weekend'.format(names[i]))
plt.show()

plt.figure(figsize =  (25, 50))
n = 5
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(5):
    plt.subplot(5,1,i+1)
    sns.boxplot(x = df.index.weekday, 
                    y = df[names[i]])
    plt.xticks(ticks = [0,1,2,3,4,5,6],labels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.title('Weekday Time Series Boxplots of {}'.format(names[i]))


plt.show()

plt.figure(figsize =  (25, 50))
n = 5
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(5):
    plt.subplot(5,1,i+1)
    sns.boxplot(x = df.index.month, 
                    y = df[names[i]])
    plt.xticks(ticks = list(range(12)), labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    plt.title('Monthly Time Series Boxplots of {}'.format(names[i]))


plt.show()


# In[24]:


plt.figure(figsize =  (25, 50))
n = 5
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(5):
    plt.subplot(5,1,i+1)
    sns.boxplot(x = df.index.year, 
                    y = df[names[i]])
    plt.xticks(ticks = list(range(5)), labels = ['2006', '2007', '2008', '2009', '2010'])
    plt.title('Yearly Time Series Boxplots of {}'.format(names[i]))


plt.show()


# In[25]:


seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
df['season'] = df.index.month.map(seasons)


# In[26]:


df['year'] = df.index.year


# In[27]:


# plt.figure(figsize =  (25, 50))
# n = 5
# names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
# for i in range(5):
#     plt.subplot(5,1,i+1)
#     sns.boxplot(x = df['season'], 
#                 y = df[names[i]],
#                 hue = df['year'],
#                 order = ['Winter', 'Spring', 'Summer', 'Autumn'],
#                 hue_order = sorted(df['year'].unique()))
#     plt.title('Seasonal Time Series Boxplots of {} by Year'.format(names[i]))
# plt.show()
plt.figure(figsize =  (25, 50))
n = 5
names = ['Global_active_power','total_energy_consumption', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for i in range(5):
    plt.subplot(5,1,i+1)
    sns.boxplot(x = df['season'], 
                y = df[names[i]],
                order = ['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.title('Seasonal Time Series Boxplots of {} by Year'.format(names[i]))
plt.show()


# In[28]:


y = df.iloc[:, -1]
X = df.iloc[:,:-1]


# Trends

# In[46]:


y.plot();


# In[56]:


sns.heatmap(df.corr(numeric_only=True), annot=True);
plt.title('Correlation plot of Variables')


# In[34]:


from statsmodels.tsa.seasonal import STL, seasonal_decompose


# In[42]:


stl = STL(y, period=30)
res = stl.fit()
fig =  res.plot()
plt.show()


# In[72]:


def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)
    plt.show()

stl = STL(y, period=12, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
res_non_robust = STL(y, period=12, robust=False).fit()
add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"])


# In[41]:


result = seasonal_decompose(y, model='additive')

# Plot the decomposed components
result.plot() 
plt.show()




# Create a figure with specified size
fig = plt.figure(figsize=(22,20))
# Adjust the subplot spacing
fig.subplots_adjust(hspace=1)

# Create first subplot
ax1 = fig.add_subplot(5,1,1)
# Plot the resampled mean of Global_active_power over day with different color
ax1.plot(df['Global_active_power'].resample('D').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax1.set_title('Mean Global active power resampled over day')
# Set major tick parameters for the subplot
ax1.tick_params(axis='both', which='major')

# Create second subplot
ax2 = fig.add_subplot(5,1,2, sharex=ax1)
# Plot the resampled mean of Global_active_power over week with different color
ax2.plot(df['Global_active_power'].resample('W').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax2.set_title('Mean Global active power resampled over week')
# Set major tick parameters for the subplot
ax2.tick_params(axis='both', which='major')

# Create third subplot
ax3 = fig.add_subplot(5,1,3, sharex=ax1)
# Plot the resampled mean of Global_active_power over month with different color
ax3.plot(df['Global_active_power'].resample('M').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax3.set_title('Mean Global active power resampled over month')
# Set major tick parameters for the subplot
ax3.tick_params(axis='both', which='major')

# Create third subplot
ax4  = fig.add_subplot(5,1,4, sharex=ax1)
# Plot the resampled mean of Global_active_power over month with different color
ax4.plot(df['Global_active_power'].resample('Q').mean(),linewidth=1, color='purple')
# Set the title for the subplot
ax4.set_title('Mean Global active power resampled over quarter')
# Set major tick parameters for the subplot
ax4.tick_params(axis='both', which='major')


# Create third subplot
ax5  = fig.add_subplot(5,1,5, sharex=ax1)
# Plot the resampled mean of Global_active_power over month with different color
ax5.plot(df['Global_active_power'].resample('A').mean(),linewidth=1, color='purple')
# Set the title for the subplot
ax5.set_title('Mean Global active power resampled over year')
# Set major tick parameters for the subplot
ax5.tick_params(axis='both', which='major')


# In[54]:


# Create a figure with specified size
fig = plt.figure(figsize=(22,20))
# Adjust the subplot spacing
fig.subplots_adjust(hspace=1)

# Create first subplot
ax1 = fig.add_subplot(5,1,1)
# Plot the resampled mean of total_energy_consumption over day with different color
ax1.plot(df['total_energy_consumption'].resample('D').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax1.set_title('Mean Total energy consumption resampled over day')
# Set major tick parameters for the subplot
ax1.tick_params(axis='both', which='major')

# Create second subplot
ax2 = fig.add_subplot(5,1,2, sharex=ax1)
# Plot the resampled mean of total_energy_consumption over week with different color
ax2.plot(df['total_energy_consumption'].resample('W').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax2.set_title('Mean Total energy consumption resampled over week')
# Set major tick parameters for the subplot
ax2.tick_params(axis='both', which='major')

# Create third subplot
ax3 = fig.add_subplot(5,1,3, sharex=ax1)
# Plot the resampled mean of total_energy_consumption over month with different color
ax3.plot(df['total_energy_consumption'].resample('M').mean(), linewidth=1, color='purple')
# Set the title for the subplot
ax3.set_title('Mean Total energy consumption resampled over month')
# Set major tick parameters for the subplot
ax3.tick_params(axis='both', which='major')

# Create third subplot
ax4  = fig.add_subplot(5,1,4, sharex=ax1)
# Plot the resampled mean of total_energy_consumption over month with different color
ax4.plot(df['total_energy_consumption'].resample('Q').mean(),linewidth=1, color='purple')
# Set the title for the subplot
ax4.set_title('Mean Total energy consumption resampled over quarter')
# Set major tick parameters for the subplot
ax4.tick_params(axis='both', which='major')


# Create third subplot
ax5  = fig.add_subplot(5,1,5, sharex=ax1)
# Plot the resampled mean of total_energy_consumption over month with different color
ax5.plot(df['total_energy_consumption'].resample('A').mean(),linewidth=1, color='purple')
# Set the title for the subplot
ax5.set_title('Mean Total energy consumption resampled over year')
# Set major tick parameters for the subplot
ax5.tick_params(axis='both', which='major')


# In[58]:


for col in df.columns:
# Create a figure with specified size
    fig = plt.figure(figsize=(22,20))
    # Adjust the subplot spacing
    fig.subplots_adjust(hspace=1)

    # Create first subplot
    ax1 = fig.add_subplot(5,1,1)
    # Plot the resampled mean of total_energy_consumption over day with different color
    ax1.plot(df[col].resample('D').mean(), linewidth=1, color='purple')
    # Set the title for the subplot
    ax1.set_title('Mean ' + col + ' resampled over day')
    # Set major tick parameters for the subplot
    ax1.tick_params(axis='both', which='major')

    # Create second subplot
    ax2 = fig.add_subplot(5,1,2, sharex=ax1)
    # Plot the resampled mean of total_energy_consumption over week with different color
    ax2.plot(df[col].resample('W').mean(), linewidth=1, color='purple')
    # Set the title for the subplot
    ax2.set_title('Mean ' + col + ' resampled over week')
    # Set major tick parameters for the subplot
    ax2.tick_params(axis='both', which='major')

    # Create third subplot
    ax3 = fig.add_subplot(5,1,3, sharex=ax1)
    # Plot the resampled mean of total_energy_consumption over month with different color
    ax3.plot(df[col].resample('M').mean(), linewidth=1, color='purple')
    # Set the title for the subplot
    ax3.set_title('Mean ' + col + ' resampled over month')
    # Set major tick parameters for the subplot
    ax3.tick_params(axis='both', which='major')

    # Create third subplot
    ax4  = fig.add_subplot(5,1,4, sharex=ax1)
    # Plot the resampled mean of total_energy_consumption over month with different color
    ax4.plot(df[col].resample('Q').mean(),linewidth=1, color='purple')
    # Set the title for the subplot
    ax4.set_title('Mean ' + col + ' resampled over quarter')
    # Set major tick parameters for the subplot
    ax4.tick_params(axis='both', which='major')


    # Create third subplot
    ax5  = fig.add_subplot(5,1,5, sharex=ax1)
    # Plot the resampled mean of total_energy_consumption over month with different color
    ax5.plot(df[col].resample('A').mean(),linewidth=1, color='purple')
    # Set the title for the subplot
    ax5.set_title('Mean ' + col + ' resampled over year')
    # Set major tick parameters for the subplot
    ax5.tick_params(axis='both', which='major')


# In[73]:


n = 7
for i, col in enumerate(df.columns):
    plt.subplot(4, 2, i+1)
    stl = STL(df[col], period=12, robust=True)
    res_robust = stl.fit()
    fig = res_robust.plot()
    res_non_robust = STL(df[col], period=12, robust=False).fit()
    add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"]);


# In[ ]:




