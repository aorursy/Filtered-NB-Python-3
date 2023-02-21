#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd




import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()




df= pd.read_csv('../input/911.csv')




df.info()




df.head()




df['zip'].value_counts().head(5)




df['twp'].value_counts().head(5)




df['title'].nunique()






df['reason']=df['title'].apply(lambda x: x.split(':')[0])
df




df['reason'].value_counts()




df['reason'].iplot(kind='histogram', size=8)




type(df['timeStamp'].iloc[0])




df['timeStamp']=pd.to_datetime(df['timeStamp'])




df['hour']= df['timeStamp'].apply( lambda x: x.hour)
df['month']=df['timeStamp'].apply(lambda x: x.month)
df['dayofweek']=df['timeStamp'].apply(lambda x:x.dayofweek)




dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}




df['dayofweek']=df['dayofweek'].map(dmap)
df




sns.countplot(x='dayofweek',hue='reason',data=df,palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0)




sns.countplot(x='month',hue='reason',data=df,palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)




callsByMonth=df.groupby('month').count()
callsByMonth.head()




callsByMonth['reason'].iplot(kind='line')




df['date']=df['timeStamp'].apply(lambda x: x.date())




callsByDate=df.groupby('date').count()['reason']
callsByDate.iplot(kind='line',size=8)




df[df['reason']=='Traffic'].groupby('date').count()['twp'].iplot(kind='line',size=8,title='traffic')




df[df['reason']=='Fire'].groupby('date').count()['twp'].iplot(kind='line',size=8,title='Fire')




df[df['reason']=='EMS'].groupby('date').count()['twp'].iplot(kind='line',size=8,title='EMS')




newData= df.groupby(['dayofweek','hour']).count()['reason'].unstack()
newData




newData.iplot(kind='heatmap',size=8,colorscale='RdYlBu',title='Total Calls received by hours')




newData2= df.groupby(['dayofweek','month']).count()['reason'].unstack()
newData2




newData2.iplot(kind='heatmap',size=8,colorscale='RdYlBu',title='Total Calls received by days and months')






