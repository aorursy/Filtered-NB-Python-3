#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




df = pd.read_excel('/kaggle/input/inflation-for-nigeria-and-turkey/Inflation consumer prices (annual ).xlsx')
df.head()




get_ipython().system('pip install dabl')
import dabl




dabl.detect_types(df)




dabl.plot(df, target_col="Country Name")




get_ipython().system('pip install autoviz')
import numpy as np
import pandas as pd 
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()




df = AV.AutoViz(filename="",sep=',', depVar='Country Name', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=30)




#Code by Olga Belitskaya https://www.kaggle.com/olgabelitskaya/sequential-data/comments
from IPython.display import display,HTML
c1,c2,f1,f2,fs1,fs2='#2B3A67','#42a7f5','Akronim','Smokum',30,15
def dhtml(string,fontcolor=c1,font=f1,fontsize=fs1):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family="""\
    +font+"""&effect=3d-float';</style>
    <h1 class='font-effect-3d-float' style='font-family:"""+\
    font+"""; color:"""+fontcolor+"""; font-size:"""+\
    str(fontsize)+"""px;'>%s</h1>"""%string))
    
    
dhtml('Programming is more than an important practical art. It is also a gigantic undertaking in the foundations of knowledge, Grace Hopper quote' )

