#!/usr/bin/env python
# coding: utf-8
#Author: Sushanth Keshav
#Last Updated : 09.Feb.2020
#Tool to find the Steady state using a graphical approach

''''
The tool helps to find the steady state through the following process:
1. Data Acquisition : ENsure only the 5 log files exist : mini8.TXT \n ,mini8_OP.TXT \n ,TE1-16.TXT \n ,TE17-32.TXT \n ,UI_Vakuum.TXT
2. Plot the P HK1 data and save the image
3. User has to open the image file saved in the folder and check for the constant PHK1 and must note down the starting values for the indexes(X-Axes value) 
4. Later user inputs the required steady states and their corresponding index start values (Note that markings on x axes correspond to 1 hour duration)
5. Steady state Median Value is calculated and later is tored in a .TXT file

''''

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import glob
import mplcursors


# In[ ]:


print("*********Please ensure that the current folder consists only the following text files: mini8.TXT \n ,mini8_OP.TXT \n ,TE1-16.TXT \n ,TE17-32.TXT \n ,UI_Vakuum.TXT \n *********************")
file_list = [f for f in glob.glob("*.TXT")]
#"Choose the file which ENDS with_Vakuum.txt" 
matching = [s for s in file_list if "UI_Vakuum" in s]
#Remove the Vakumm.TXT file and insert it in 1st position
file_list.remove(matching[0])
#insert it in the position 1
file_list.insert(0,matching[0])


# In[ ]:


#Using dataframes to collect data and consilidate it

df_list = [pd.read_csv(filename, sep="\t",skiprows=6,encoding = 'unicode_escape') for filename in file_list]
df = df_list[0]
for df_ in df_list[1:]:
    df = df.merge(df_, on=['Datum', 'Uhrzeit'], how='left')
df=df.stack().str.replace(',','.').unstack() #Replace all the , with .


# In[ ]:


#from the dataframe to Numpy Array -- importing data except the date and time
arr =np.array(df.iloc[:,2:].astype(float))
rows,columns = arr.shape


# In[ ]:
#'''' For the plotting initially we check if there are outliers, if they exist: we use the Quartile range to set our plot axes for legible reading''''

#Plotting the P HK1 by exclusding the outliers
y_sorted = np.sort(arr[:,5])

#Finding first quartile and third quartile
q1, q3= np.percentile(y_sorted,[25,75])

#Find the IQR which is the difference between third and first quartile
iqr = q3 - q1

#Upper bound outlier check : Reason why we choose 60 is: 60 index counts = 10mins : Assume that outlier duration is not more than 10mins
if np.std(y_sorted[-60:]) >1:
    print('its going out of bound')
    upper_bound = q3 +(5 * iqr) 
else:
    upper_bound = y_sorted[-1]

#Lower Bound outlier check
if np.std(y_sorted[:60]) > 1:
    lower_bound = q1 -(3 * iqr)
else:
    lower_bound = y_sorted[0]


#Plot the P HK1 
fig, ax = plt.subplots(figsize = (40,20),dpi=220)

#Plotting the P HK1 on y axis
ax.plot(arr[:,5])
#ax.set_ylim(np.amin(arr[:,5]),np.amax(arr[:,5])) --- doesn#t work if there are outlier
ax.set_ylim(lower_bound,upper_bound)

#Major axis indexing for 1 hr intervals while the minor axis for 30mins -- cooresponds to marking
ax.xaxis.set_major_locator(MultipleLocator(360))
ax.xaxis.set_minor_locator(MultipleLocator(180))

ax.set_title("P HK1[W] v/s Index Values to Choose",fontsize=38)

ax.set_xlabel("Index Values to Choose", fontsize=38)
ax.set_ylabel("P HK1[W]", fontsize= 38)

ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

plt.xticks(rotation=90) #Rotate the x axes labels

fig.savefig(r'{fn[0]}_SteadyStatePlotFigure.png'.format(fn=file_list))

#plt.show()

# In[ ]:


# Taking user input for the steady state computation..Need to refer plot and input the data

print("\n The P HK1[W] graph has been plotted and saved in the directory.\n Please Open and type the Index values to compute the Steady States. \n")
print("\n NOTE: The markings correspond to 1-hour intervals!!! \n")
steady_state_value= []
index_values= int(input("How many Steady State Values do you want to compute??"))
for w in range(index_values):
    variable = int(input(str("Type the start index of the Steady State Number ")+str(w+1)+str(": ")))
    steady_state_value.append(variable)


# In[ ]
''''
The Observation recording time duration = 10seconds i.e, time difference between index1 and index2 = 10seconds 
So, if for every 10 seconds one recording is done, for 1 hour we would have 360 recordings. Here recording(or rows) are seen as index counts
    Implies : 1 hour                       = 3,600 seconds 
              Observation index Count for 1 hour = 3,600/10 = 360 index counts
''''

# After user gives the start index value, +360 values need to be gruped for getting 1 hiur steady state because 3600 sec=1hr and the recording interval is 10 sec so 360 observations correspond to 1hour
correct_steady = []
for corr in range(len(steady_state_value)):
    duration = arr[steady_state_value[corr]:steady_state_value[corr]+360,:]
    correct_steady.append(duration)


# In[ ]:


#computing the median data
median_data = np.zeros((len(correct_steady),(correct_steady[0].shape[1])))
for m in range(len(correct_steady)):
    median_data[m,:] = np.median(correct_steady[m], axis=0)


# In[ ]:


# Writing the median data into a txt file 
# The median data is connected with the date and start-end time of steady state
steady_state_duration = []
for count,element in enumerate(steady_state_value):
    steady_state_duration.append(np.array([df.iloc[element,0],df.iloc[element,1], df.iloc[element+360,1]]))
    #steady_state_duration.append(np.array([df.iloc[correct_end_index[count]-min_time,0],df.iloc[correct_end_index[count]-min_time,1], df.iloc[correct_end_index[count],1]]))
df_cols = df.columns.tolist()
new_df = pd.DataFrame(steady_state_duration, columns=['Datum', 'SteadyState_Start', 'Steady_state_end'])
new_df2 = pd.DataFrame(median_data, columns = df_cols[2:])
steady_state_df = pd.concat([new_df,new_df2],axis=1)
steady_state_df.to_csv(r'{fn[0]}_1hrSteadyStateMedianData1hr.txt'.format(fn=file_list), header=True, index=None, sep='\t', mode='w')


# In[ ]:




