#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import folium
from geopy.distance import distance
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
plt.style.use('seaborn-whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv("../input/mahadir_driving_data.csv")




df.info()




df.head()




df.groupby(['Data Source','Data'], as_index=False).agg({'Timestamp':'count'})




df['timestamp'] = pd.to_datetime(df['Timestamp'])
df['gp'] = df['Data Source'] + " (" + df['Data'] + ")"




df = df.drop(columns=['Timestamp'])




df.head()




df = df.pivot_table(index='timestamp', columns='gp', values=['Raw Value'], aggfunc='first')




df.columns = df.columns.droplevel()




df = df.reset_index()




df.columns




len(df)




df = df.dropna(subset=['GPS (Latitude)', 'GPS (Longitude)'])




len(df)




df[['timestamp','GPS (Latitude)','GPS (Longitude)','GPS (Speed (estimated))']].head()




df['GPS (Speed (estimated))'] = df['GPS (Speed (estimated))'].astype('float64')
df['Altimeter (Barometer) (Relative Altitude)'] = df['Altimeter (Barometer) (Relative Altitude)'].astype('float64')
df['GPS (Latitude)'] = df['GPS (Latitude)'].astype('float64')
df['GPS (Longitude)'] = df['GPS (Longitude)'].astype('float64')

df['Gyrometer (raw) (x)'] = df['Gyrometer (raw) (x)'].astype('float64')
df['Gyrometer (raw) (y)'] = df['Gyrometer (raw) (y)'].astype('float64')
df['Gyrometer (raw) (z)'] = df['Gyrometer (raw) (z)'].astype('float64')

df['Acceleration (total) (x)'] = df['Acceleration (total) (x)'].astype('float64')
df['Acceleration (total) (y)'] = df['Acceleration (total) (y)'].astype('float64')
df['Acceleration (total) (z)'] = df['Acceleration (total) (z)'].astype('float64')




print(df['GPS (Latitude)'].min(),df['GPS (Latitude)'].max())
print(df['GPS (Longitude)'].min(),df['GPS (Longitude)'].max())




# convert ms/s to km/h
df['km/h'] = df['GPS (Speed (estimated))'] * 3.6




# our plotting utilities
def plot(vals, title="", xlabel="", ylabel='', colorbar=None):
    fig = plt.figure(figsize=(14,3))
    ax = plt.subplot(111)
    for val in vals:
        label = ''
        c = 'b'
        if 'label' in val:
            label = val['label']
        if 'c' in val:
            c = val['c']
        if 'x' in val:
            x = val['x']
        else:
            x = [x for x in range(len(val['y']))]
        ax.plot(x,val['y'], label=label,c=c)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if colorbar is not None:
        cax, _ = matplotlib.colorbar.make_axes(ax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=colorbar['cmap'], norm=colorbar['norm'])
    plt.show()




vals = [
        {'x':df['timestamp'].values,'y':df['Gyrometer (raw) (x)'].values, 'label':'x', 'c':'r'},
        {'x':df['timestamp'].values,'y':df['Gyrometer (raw) (y)'].values, 'label':'y', 'c':'y'},
        {'x':df['timestamp'].values,'y':df['Gyrometer (raw) (z)'].values, 'label':'z', 'c':'b'},
       ]
plot(vals, title='Gyro', xlabel='timestamp', ylabel='rad/s')




vals = [
        {'x':df['timestamp'].values,'y':df['Acceleration (total) (x)'].values, 'label':'x', 'c':'r'},
        {'x':df['timestamp'].values,'y':df['Acceleration (total) (y)'].values, 'label':'y', 'c':'y'},
        {'x':df['timestamp'].values,'y':df['Acceleration (total) (z)'].values, 'label':'z', 'c':'b'},
       ]
plot(vals, title='Accelerometer (total)', xlabel='timestamp', ylabel='m/s2')




vals = [
        {'x':df['timestamp'].values,'y':df['Altimeter (Barometer) (Relative Altitude)'].values, 'label':'x', 'c':'r'}
       ]
plot(vals, title='Altimeter (Barometer) (Relative Altitude)', xlabel='timestamp', ylabel='meter')




df['Altimeter (Barometer) (Relative Altitude)'].describe()




#3.99950996507332,101.152279800398 
#4.50031313114287,101.310494793673
norm = matplotlib.colors.Normalize(vmin=df['Altimeter (Barometer) (Relative Altitude)'].min(), vmax=df['Altimeter (Barometer) (Relative Altitude)'].max())
cmap = matplotlib.cm.get_cmap('jet')
m = folium.Map(location=[4.50031313114287,101.310494793673],
    zoom_start=10, width=1200, height=480,  tiles='Stamen Terrain')

# Plot coordinates
i = 0
for i in range(len(df)):
    rgba = cmap(norm(df.iloc[i]['Altimeter (Barometer) (Relative Altitude)']))
    color = matplotlib.colors.rgb2hex(rgba)
    # mark only every 5 point
    if i%5==0:
        _ = folium.CircleMarker(location=[df.iloc[i]['GPS (Latitude)'], df.iloc[i]['GPS (Longitude)']], radius=1,color=color, fill_color=color).add_to(m)
    i += 1




m




vals = [
        {'x':df['timestamp'].values,'y':df['km/h'].values, 'label':'x', 'c':'g'}
       ]
plot(vals, title='Speed', xlabel='timestamp', ylabel='km/h')




df_temp = df[['GPS (Latitude)','GPS (Longitude)','km/h']].copy()
df_temp = df_temp.rename(columns={'GPS (Latitude)':'lat','GPS (Longitude)':'long'})




get_ipython().run_cell_magic('time', '', "# find nearest lat long from the Point of Interest\npoi = (4.3429785,101.2341151)\nnearest = 100000 # just randomly huge number\nposition = 0 # the index position\ncoord = None\nfor i in range(len(df_temp)):\n    if np.isnan(df_temp.iloc[i]['lat']):\n        continue\n    latlong = (df_temp.iloc[i]['lat'],df_temp.iloc[i]['long'])\n    d = distance(poi, latlong).m\n    if d < nearest:\n        nearest = d\n        position = i\n        coord = latlong\nprint(nearest,position, coord)")




# since the POI KM is > than AES coordinate we'll need to work backward
df_temp['lat_shift'] = df_temp['lat'].shift(-1)
df_temp['long_shift'] = df_temp['long'].shift(-1)




get_ipython().run_cell_magic('time', '', "df_temp['km'] = 0\ncurrent = 312\ndf_temp.loc[position,'km']=current\nfor i in range((position-1)):\n    x = (position-1)-i\n    data = df_temp.iloc[i]\n    # lat long of next\n    latlong1 = (data['lat_shift'],data['long_shift'])\n    # lat long of current\n    latlong2 = (data['lat'],data['long'])\n    # distance in meter\n    d = distance(latlong2, latlong1).m\n    #print(latlong1,latlong2)\n    current = current - (d/1000)\n    df_temp.loc[x,'km']=current")




# 98 km/h is under the limit of 110km/h 
df_temp[(df_temp['km']>299) & (df_temp['km']<300)]['km/h'].max()




# summary statistics of the speed
df['km/h'].describe()




df['acceleration'] = (df['GPS (Speed (estimated))']-df['GPS (Speed (estimated))'].shift())/(df['timestamp']-df['timestamp'].shift()).dt.seconds




df['acceleration'].describe()




print("Percentage of acceleration > 2:", len(df[np.abs(df['acceleration']) > 2])/len(df) * 100)




print("Percentage of acceleration > 3:", len(df[np.abs(df['acceleration']) > 3])/len(df) * 100)




df['seconds'] = (df['timestamp']-df['timestamp'].shift()).dt.seconds
df['seconds'] = df['seconds'].cumsum()
df.loc[0,'seconds'] = 0




get_ipython().run_cell_magic('time', '', 'norm = matplotlib.colors.Normalize(vmin=df[\'acceleration\'].min(), vmax=df[\'acceleration\'].max())\ncmap = matplotlib.cm.get_cmap(\'jet\')\nz = []\nfor i in range(len(df)):\n    rgba = cmap(norm(df.iloc[i][\'acceleration\']))\n    z.append(matplotlib.colors.rgb2hex(rgba))\nfig = plt.figure(figsize=(15,5))\nax = plt.subplot(111)\nax.scatter(df[\'seconds\'].values, df[\'GPS (Speed (estimated))\'].values, c=z)\nplt.title("Speed and Acceleration")\nplt.ylabel("m/s")\nplt.xlabel("seconds")\ncax, _ = matplotlib.colorbar.make_axes(ax)\ncbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)\nplt.show()')




get_ipython().run_cell_magic('time', '', '# model acceleration, constant and deceleration\nacc = df[df[\'acceleration\']>0]\nconst = df[df[\'acceleration\']==0]\ndecc = df[df[\'acceleration\']<0]\nfig = plt.figure(figsize=(15,5))\nax = plt.subplot(111)\nax.plot(\n    acc[\'seconds\'].values, acc[\'GPS (Speed (estimated))\'].values, \'ro\',\n    const[\'seconds\'].values, const[\'GPS (Speed (estimated))\'].values, \'go\',\n    decc[\'seconds\'].values, decc[\'GPS (Speed (estimated))\'].values, \'bo\',\n)\nplt.title("Speed and Acceleration")\nplt.ylabel("m/s")\nplt.xlabel("seconds")\nplt.legend([\'Accelerate\',\'Constant\', \'Decelerate\'])\nplt.show()')




# how many stops?
index = []
start = None
end = None
for k,v in df[df['GPS (Speed (estimated))']<=0].iterrows():
    if k-1 == end:
        #replace end
        end = k
    elif len(index) > 0:
        # no longer continuous
        index[-1]['end'] = end
        start = None
        end = None
        
    if start is None:
        start = k
        end = k
        index.append({'start':start, 'end':end})




# a prolong continuous stop and moving can indicate in a bad traffic
index




print("Total stops", len(index))




get_ipython().run_cell_magic('time', '', 'data = df[(df[\'seconds\']>2110) & (df[\'seconds\']<2190)]\nz = []\nfor k,v in data.iterrows():\n    if v[\'acceleration\'] > 0:\n        z.append(True)\n    else:\n        z.append(False)\n    \nfig = plt.figure(figsize=(15,5))\nax = plt.subplot(111)\n#ax.scatter(data[\'seconds\'].values, data[\'GPS (Speed (estimated))\'].values, c=z)\nax.plot(\n    data[np.array(z)][\'seconds\'], data[np.array(z)][\'GPS (Speed (estimated))\'], \'ro\',\n    data[~np.array(z)][\'seconds\'], data[~np.array(z)][\'GPS (Speed (estimated))\'], \'bo\',\n)\nplt.title("Speed and Acceleration")\nplt.ylabel("m/s")\nplt.xlabel("seconds")\nplt.legend([\'Accelerate\', \'Decelerate\'])\nplt.show()')




# at index 2179 is at Bidor plaza toll
pos = 2179
start = 0
for i in range(pos):
    backward = pos - (i+1) 
    # find the first positive acceleration
    if df.iloc[backward]['acceleration'] > 0:
        start = backward+1 
        print(start)
        break




progressive_df = pd.DataFrame(
    [
        {'speed_mph':10,'second_to_stop':8, 'feet_to_stop':86},
        {'speed_mph':20,'second_to_stop':11, 'feet_to_stop':207},
        {'speed_mph':30,'second_to_stop':14, 'feet_to_stop':372},
        {'speed_mph':40,'second_to_stop':17, 'feet_to_stop':581},
        {'speed_mph':50,'second_to_stop':20, 'feet_to_stop':851},
        {'speed_mph':60,'second_to_stop':24, 'feet_to_stop':1262},
    ]
)
progressive_df['speed_m/s'] = progressive_df['speed_mph']/2.237
progressive_df['km/h'] = progressive_df['speed_mph']*1.609
progressive_df['metre_to_stop'] = progressive_df['feet_to_stop']/3.281




progressive_df




# do some extrapolation
p = np.polyfit(progressive_df['speed_m/s'],progressive_df['second_to_stop'],2) 




f = np.poly1d(p)
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.plot(progressive_df['speed_m/s'], progressive_df['second_to_stop'], 'ro', label="Data")
plt.plot(progressive_df['speed_m/s'],f(progressive_df['speed_m/s']), 'b-',label="Polyfit")
plt.show()




latlong1 = (df.iloc[start]['GPS (Latitude)'],df.iloc[start]['GPS (Longitude)'])
latlong2 = (df.iloc[pos]['GPS (Latitude)'],df.iloc[pos]['GPS (Longitude)'])
d = distance(latlong2, latlong1).m




print("Starting speed at braking event",df.iloc[start]['km/h'],"km/h")
print("Time taken to complete stop",df.iloc[pos]['seconds'] - df.iloc[start]['seconds'], "seconds")
print("Distance taken to complete stop is", d, "metre")
print("According to Progressive on average this should be approximately", f(df.iloc[start]['GPS (Speed (estimated))']),'seconds')




len(df)




#3.99950996507332,101.152279800398 
#4.50031313114287,101.310494793673
norm = matplotlib.colors.Normalize(vmin=df['km/h'].min(), vmax=df['km/h'].max())
cmap = matplotlib.cm.get_cmap('jet')
m = folium.Map(location=[4.50031313114287,101.310494793673],
    zoom_start=10, width=1200, height=480,  tiles='Cartodb dark_matter')

# Plot coordinates
i = 0
for i in range(len(df)):
    rgba = cmap(norm(df.iloc[i]['km/h']))
    color = matplotlib.colors.rgb2hex(rgba)
    # mark only every 5 point
    if i%5==0:
        _ = folium.CircleMarker(location=[df.iloc[i]['GPS (Latitude)'], df.iloc[i]['GPS (Longitude)']], radius=1,color=color, fill_color=color).add_to(m)
    i += 1




get_ipython().run_cell_magic('time', '', 'm')




ls






