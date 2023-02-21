#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

plt.style.use('bmh')




import plotly.graph_objs as go
import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot
from plotly import tools
init_notebook_mode()




def annt_bar(ax, ort, fmt='%d'):
    if ort == 'h':
        for p in ax.patches:
            ax.annotate(fmt%p.get_width(), (p.get_width()+ax.get_xlim()[1]*0.02, p.get_y()))
    elif ort == 'v':
        for p in ax.patches:
            ax.annotate(fmt%p.get_height(), (p.get_x(), p.get_height()+ax.get_ylim()[1]*0.02))




path = '../input/lending-club-loan-data/loan.csv'




with open(path) as handle:
    lines = sum(1 for l in handle)




skips = np.sort(np.random.choice((1, lines), size=lines-1-100000))
df = pd.read_csv(path, skiprows=skips, low_memory=False)




df['loan_status'].value_counts().plot(kind='barh')




## for convenience, I will just pay attention to the ended loan
df = df[~df.loan_status.isin(['Current', 'Issued'])] 
df.info()




df['loan_status'].value_counts().plot(kind='barh')




ok_loan_status = [#"Current", 
                  "Fully Paid",
                  #"In Grace Period", 
                  "Does not meet the credit policy. Status:Fully Paid"]

# NOTE: https://www.collinsdictionary.com/dictionary/english/grace-period




df['ok_loan'] = (df.loan_status.isin(ok_loan_status)).astype(int)




loan_year = df.groupby(pd.to_datetime(df.issue_d).dt.year)['ok_loan'].value_counts().unstack()




data = []


# Pie
asp = 0.3
cr = 0.038
dm_a, dm_b = 0.5-cr, 0.5+cr
step = dm_a / loan_year.shape[0]

for i, row in enumerate(loan_year.iterrows()):
    dm = np.array([dm_a-i*step, dm_b+i*step])
    trace = go.Pie(
            values = row[1].values,
            domain = {"x": dm*asp, "y": dm},
            labels = ["Risk Loan", "OK Loan"],
            hole = 1 - 0.96/(cr/step+i),
            name = str(row[0]),
            hoverinfo = "label+percent+name+text",
            rotation = 90, 
            textposition = "inside",
            opacity = min(3*row[1].values[0]/sum(row[1].values), 1.0),
            marker = dict(colors=['red','lightblue']),
            showlegend=False
        )
    data.append(trace)

    
# Bar   
data.append(go.Bar(x=loan_year.index, 
                   y=loan_year.iloc[:,0], 
                   name='OK Loan', 
                   marker=dict(color='lightblue')))
data.append(go.Bar(x=loan_year.index, 
                   y=loan_year.iloc[:,1], 
                   name='Risk Loan', 
                   marker=dict(color='red')))

    
fig = dict(
   data=data,
   layout=dict(title="Risk Loan Rate 2007~2018", 
               xaxis=dict(title='Amount', 
                          domain=[0.35, 1]),
               yaxis=dict(title='Count'),
               width=1000, 
               height=400)
)

iplot(fig)




lo = {}
pprm = {}
lo['hist'] = dict(bargap=0.2, bargroupgap=0.1)
pprm['hist'] = dict(nbinsx=20, opacity=0.85)




tr1 = go.Histogram(x=df[df.ok_loan==1]['loan_amnt'], 
                   name='OK Loan',   
                   marker=dict(color='lightblue'), 
                   **pprm['hist'])

tr2 = go.Histogram(x=df[df.ok_loan==0]['loan_amnt'], 
                   name='Risk Loan', 
                   marker=dict(color='red'), 
                   **pprm['hist'])

fig = dict(data=[tr1, tr2], layout=dict(title='Loan Amount Loan Status',
                                        xaxis=dict(title='Amount'),
                                        yaxis=dict(title='Count'), 
                                        height=300))
iplot(fig)




tr1 = go.Histogram(x=df[df.ok_loan==1]['loan_amnt'], histnorm='percent', name='OK Loan',   marker=dict(color='lightblue'), **pprm['hist'])
tr2 = go.Histogram(x=df[df.ok_loan==0]['loan_amnt'], histnorm='percent', name='Risk Loan', marker=dict(color='red'), **pprm['hist'])

fig = dict(data=[tr1, tr2], layout=dict(title='Loan Amount Loan Status (Percent)',
                                        xaxis=dict(title='Amount'),
                                        yaxis=dict(title='Count'), 
                                        height=300))
iplot(fig)




mths_cols = df.columns[df.columns.str.contains('mths')]
df[mths_cols].info()




since_cols = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']
df[since_cols].plot.kde(xlim=(0, 130), figsize=(15, 3)) # manual xlim ;P




fig = tools.make_subplots(rows=3, cols=1, subplot_titles=since_cols)
fig.layout.update(title='Inspect Months')
                  

for i, col in enumerate(since_cols):
    tr1 = go.Histogram(x=df[df.ok_loan==1][col], name='OK Loan', histnorm='percent', 
                       marker=dict(color='lightblue'), **pprm['hist'])
    tr2 = go.Histogram(x=df[df.ok_loan==0][col], name='Risk Loan', histnorm='percent', 
                       marker=dict(color='red'), **pprm['hist'])
    fig.append_trace(tr1, i+1, 1)
    fig.append_trace(tr2, i+1, 1)

iplot(fig)




df['last_pymnt_d_from_issue'] = (pd.to_datetime(df.last_pymnt_d) - pd.to_datetime(df.issue_d)).dt.days
df['last_credit_pull_d_from_issue'] = (pd.to_datetime(df.last_credit_pull_d) - pd.to_datetime(df.issue_d)).dt.days
df['earliest_cr_line_from_issue'] = (pd.to_datetime(df.issue_d) - pd.to_datetime(df.earliest_cr_line)).dt.days 




df.loc[:,df.columns.str.contains('from_issue')].hist(bins=60, figsize=(20, 5));




def two_fac_sct(x1, x2, t1, t2):
    trans = [dict(type='groupby', groups=df.ok_loan, text=['Risk Loan', 'OK Loan'],
              styles=[dict(target=0, value=dict(marker=dict(color='red'))),
                      dict(target=1, value=dict(marker=dict(color='lightblue')))] )]
    data = [dict(type='scattergl', mode='markers', x=x1, y=x2,
             marker=dict(line=dict(width=1), size=6, opacity=0.5),
             transforms=trans)]

    fig = dict(data=data, layout=dict(xaxis=dict(title=t1), yaxis=dict(title=t2), 
                                      width=800, height=400))
    iplot(fig, validate=False)




two_fac_sct(df.last_pymnt_d_from_issue, df.last_pymnt_amnt, 
            'Last Payment from Issue Day (Days)', 
            'Last Payment Amount')




fund_col = df.columns[df.columns.str.contains('funded')]




tr1 = go.Histogram(x=df[fund_col[0]], name=fund_col[0], marker=dict(color='gold'), **pprm['hist'])
tr2 = go.Histogram(x=df[fund_col[1]], name=fund_col[1], marker=dict(color='purple'), **pprm['hist'])

fig = dict(data=[tr1, tr2], layout=dict(title='Fund Distribution',
                                        xaxis=dict(title='Amount'),
                                        yaxis=dict(title='Count'),
                                        width=600, height=300))
iplot(fig)




two_fac_sct(df.loan_amnt,df.funded_amnt, 
            'Loan Amount', 
            'Funded Amount')




part_fund = (df.loan_amnt != df.funded_amnt)
ok_target = df[(df.ok_loan==1)&part_fund]
risk_target = df[(df.ok_loan==0)&part_fund]

tr1 = go.Scatter(x=ok_target['loan_amnt'], y=ok_target['funded_amnt'], 
                   mode='markers',
                   name='OK Loan',
                   marker=dict(color='lightblue', line=dict(width=1)))
tr2 = go.Scattergl(x=risk_target['loan_amnt'], y=risk_target['funded_amnt'], 
                   mode='markers',
                   name='Risk Loan',
                   marker=dict(color='red', line=dict(width=1)))



fig = dict(data=[tr1, tr2], layout=dict(xaxis=dict(title='Loan Amount'), yaxis=dict(title='Funded Amount'), 
                                        width=500, height=400))
iplot(fig)




df['zip_code'] = df['zip_code'].apply(lambda x : str(x)[:3]+"01") # fuzzy replacement




uszip = pd.read_csv('../input/simplemaps/uszips.csv', usecols=['zip', 'lat', 'lng', 'city'], dtype={'zip':np.object})
uszip.rename(columns={'zip':'zip_code'}, inplace=True)




df = df.merge(uszip, on='zip_code') # Data Loss




df.info()
df['loan_status'].value_counts().plot(kind='barh')





target = df.groupby('city')['lat', 'lng'].mean()            .join(df.groupby('city')['ok_loan'].value_counts().unstack().fillna(0).astype(int))
 


data = []
layout = dict(title='State', width=800, height=800, legend=dict(bgcolor='rgba(0,0,0,0)'))


## Left Side
data.extend([
# Risk Loan
             go.Scattergeo(
                locationmode = 'USA-states',
                lon = target.lng+0.1, 
                lat = target.lat, 
                name = 'Risk Loan',
                mode = 'markers',
                text = target.index +' '+ target[0].astype(str),
                marker = dict(size=(target[0]/80.).apply(lambda x: min(50, max(x, 5))), color='red', symbol='x'),
                geo = 'geo1'),
# OK Loan
             go.Scattergeo(
                locationmode = 'USA-states', 
                lon = target.lng-0.1, 
                lat = target.lat, 
                name = 'OK Loan',
                mode = 'markers',
                text = target.index + ' ' + target[1].astype(str),
                marker = dict(size=(target[1]/80.).apply(lambda x: min(50, max(x, 5))), opacity = 0.3, color='skyblue'),
                geo = 'geo1')

             ])
layout['geo1'] = dict(scope='usa', projection=dict(type='albers usa'))

              
              
## Right Side
state_count = df['addr_state'].value_counts()
state_risk_rate = df.groupby('addr_state')[['ok_loan', 'lat', 'lng']].mean()
    
data.extend([go.Choropleth(
                locations = state_count.index, 
                z = state_count.values.astype(float), 
                name = 'Count',
                locationmode = 'USA-states',
                colorscale='Bluered',
                colorbar = dict(title = "Loan Issued by State", len=0.5, y=0.25),
                geo = 'geo2'),
             go.Scattergeo(
                lat = state_risk_rate.lat, 
                lon = state_risk_rate.lng, name='Risk Rate',
                text = state_risk_rate.ok_loan.apply(lambda x: '{0:.2f}%'.format(x*100)) + state_risk_rate.index,
                marker = dict(size=np.exp(state_risk_rate.ok_loan*3.5), color='lightgrey'),
                geo = 'geo2')
             ])

layout['geo2'] = dict(scope='usa', projection=dict(type='albers usa'))

layout['geo1']['domain'] = dict(x=[0, 1], y=[0.5, 1])
layout['geo2']['domain'] = dict(x=[0, 1], y=[0, 0.5])           

fig = go.Figure(data=data, layout=layout)
iplot(fig)




issue_y = pd.to_datetime(df.issue_d).dt.year.rename('issue_y')




df.emp_length.replace({'< 1 year': '0 year', '10+ years': '10 years'}, inplace=True)
df.emp_length.fillna('0 year', inplace=True) # !!!

df.emp_length = df.emp_length.apply(lambda x: x.split()[0]).astype(int)




# g = sns.boxplot(data=df, x='emp_length', y='annual_inc', hue='home_ownership', ax=ax1)
plt.figure(figsize=(20, 3))
sns.boxplot(data=df, x='emp_length', y='annual_inc', hue='home_ownership');
plt.gca().set_yscale('log')
plt.legend(loc=8, ncol=4)




plt.figure(figsize=(20, 4))
sns.boxplot(data=df, x='emp_length', y='loan_amnt', hue='home_ownership');
plt.legend(loc=8, ncol=4)




def stack_show(x_col, val_col, hue_col, r=3, ax=None, bl='zero'):
    
    data = df.pivot_table(index=x_col, 
                          values=val_col, 
                          columns=hue_col, 
                          aggfunc=np.mean, 
                          fill_value=0)

    data_r = data.rolling(r).mean()
    _ax = ax
    if _ax == None:
        _, _ax = plt.subplots(1, 1, figsize=(8, 3))
    _ax.stackplot(data_r.index, data_r.values.T, labels=data_r.columns, alpha=0.6, baseline=bl, colors=sns.color_palette("tab20"));
    _ax.legend(loc=2)
    return _ax




ax = stack_show(pd.to_datetime(df.issue_d), 'loan_amnt', 'home_ownership')
ax.set_ylabel('Loan Amount');
ax.set_xlabel('Issue Date');




_, ax = plt.subplots(1,1, figsize=(10, 3))
stack_show(pd.to_datetime(df.issue_d), 'loan_amnt', 'purpose', ax=ax, r=3, bl='wiggle')
ax.legend(bbox_to_anchor=(1, 1), loc=2, ncol=1, fontsize=7)
ax.set_ylabel('Loan Amount')




import matplotlib.gridspec as gridspec
plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(gs[1:,0:])
ax4 = plt.subplot(gs[0, 0])
ax3 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[0, 2])


df.groupby('purpose')['ok_loan'].value_counts().unstack().plot.bar(ax=ax1, width=0.8)
annt_bar(ax1, 'v')
    
df.groupby('purpose')['ok_loan'].mean().plot.barh(ax=ax2, xlim=(0.5, 1), color='purple', alpha=0.8)
annt_bar(ax2, 'h', '%.2f')
    
df.groupby('purpose')['loan_amnt'].mean().plot.barh(ax=ax3, color='green', alpha=0.3)
annt_bar(ax3, 'h', '%.2f')

df.groupby('purpose').count().id.plot.barh(color='gold', ax=ax4)
annt_bar(ax4, 'h', '%.2f')

    
ax1.set_title('Count VS Purpose')
ax2.set_title('OK Loan Rate by Purpose ')
ax3.set_title('Loan Amount by Purpose')
ax4.set_title('Count by Purpose')
plt.tight_layout()




df.initial_list_status.value_counts()




df.groupby('initial_list_status')['ok_loan'].value_counts().unstack().plot.bar(figsize=(5, 2))




df.term.value_counts()




df[df.application_type=='JOINT']['ok_loan'].value_counts()




df.grade.value_counts().sort_index().plot('bar', figsize=(5, 2));




ax = stack_show(pd.to_datetime(df.issue_d), 'loan_amnt', 'grade', r=3)
ax.set_ylabel('Loan Amount');
ax.set_xlabel('Issue Date');




ax = stack_show(pd.to_datetime(df.issue_d), 'int_rate', 'grade')
ax.set_ylabel('Interest_Rate');
ax.set_xlabel('Issue Date');




sns.lineplot(x=pd.to_datetime(df.issue_d), y='int_rate', hue='grade', data=df)




df[['installment', 'int_rate']].hist(bins=100, figsize=(10, 2));




from matplotlib import gridspec
plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1:])
ax3 = plt.subplot(gs[1, 0:])

cols = ['term', 'grade', 'sub_grade']
for col, ax in zip(cols, [ax1, ax2, ax3]):
    df.groupby(col)['ok_loan'].value_counts().unstack().plot.bar(ax=ax, width=0.9)
    for p in ax.patches:
        ax.annotate('%d'%p.get_height(), (p.get_x(), p.get_height()+ax.get_ylim()[1]*0.02), size=7)
    ax.set_ylabel('count')
plt.tight_layout()




df.groupby('ok_loan')['installment'].plot.kde(xlim=(0, df.installment.max()), figsize=(8, 3), legend=True)
plt.legend(title='OK Loan')
plt.ylabel('Installment')




df.groupby('ok_loan')['int_rate'].plot.kde(figsize=(8, 3), legend=True, xlim=(0, df.int_rate.max()))
plt.legend(title='OK Loan')
plt.ylabel('Interest Rate')




issue_d = pd.to_datetime(df.issue_d)

df.groupby([issue_d, df.ok_loan])['int_rate'].mean().unstack().fillna(method='bfill').plot(figsize=(8, 3))
plt.ylabel('Interest Rate %')




issue_y = pd.to_datetime(df.issue_d).dt.year
df.groupby([issue_y, df.ok_loan])['int_rate'].mean().unstack().fillna(method='bfill').plot(figsize=(8, 3))
plt.ylabel('Interest Rate %')




df.groupby([issue_d, df.ok_loan])['installment'].mean().unstack().fillna(method='bfill').plot(figsize=(8, 3))
plt.ylabel('Installment')




df.groupby([issue_y, df.ok_loan])['installment'].mean().unstack().fillna(method='bfill').plot(figsize=(8, 3))
plt.ylabel('Installment')




cols = ['open_acc',
        'pub_rec', 
        'total_acc']

fig, axes = plt.subplots(3, 1, figsize=(8, 6))
for col, ax in zip(cols, axes):
    df.groupby('ok_loan')[col].plot(kind='kde', ax=ax, xlim=(df[col].min(), df[col].max()))
    ax.set_title(col)
plt.tight_layout()




cols = ['revol_bal', 'revol_util']
fig, axes = plt.subplots(2, 1, figsize=(20, 5))
for col, ax in zip(cols, axes):
    df.groupby('ok_loan')[col].plot(kind='kde', ax=ax, xlim=(df[col].min(), df[col].max()))
    ax.set_title(col)





df.groupby('ok_loan')['mths_since_last_delinq'].plot(kind='kde', xlim=(0, 120), alpha=0.4, legend=True)
plt.title('mths_since_last_delinq')
plt.legend(title='OK loan')




display(df.acc_now_delinq.value_counts())
df.groupby('acc_now_delinq')['ok_loan'].mean().plot.bar()




display(df.delinq_2yrs.value_counts())
df.groupby('delinq_2yrs')['ok_loan'].mean().plot.bar()




numr_feat = [
'loan_amnt',   
'funded_amnt', 
'funded_amnt_inv',      
'out_prncp',             
'out_prncp_inv',   
'total_pymnt',              
'total_pymnt_inv',     
'last_pymnt_amnt',    
'total_rec_prncp',  
'total_rec_int',        
'total_rec_late_fee', 
'recoveries',          
'collection_recovery_fee',  
'tot_coll_amt',           
'tot_cur_bal',         
'collections_12_mths_ex_med',  
'total_rev_hi_lim',
'open_acc',
'pub_rec', 
'total_acc',
'int_rate',
'installment',
'dti',
'revol_bal',
'revol_util',
'annual_inc',
'delinq_2yrs',
'inq_last_6mths',
'mths_since_last_delinq',
'acc_now_delinq',
'lat', # !!!
'lng'  # !!!
]




plt.figure(figsize=(20, 8))
sns.heatmap(df[numr_feat].corr(), annot=True, annot_kws=dict(fontsize=8), fmt='.2f')




missing_count = df.loc[:,df.isna().any()].isna().sum(axis=0)

tr = go.Bar(x = missing_count.index, y = missing_count.values,
            marker = dict(color = '#02b3e4'))

layout = go.Layout(title='Missing Data (Not Null Count)', height=300)

fig = go.Figure([tr], layout=layout)
iplot(fig)




data = df.copy()




big_miss = [
'open_acc_6m',                    
'open_il_12m',                    
'open_il_24m',                    
'mths_since_rcnt_il',             
'total_bal_il',                   
'il_util',                       
'open_rv_12m',                    
'open_rv_24m',                   
'max_bal_bc',                    
'all_util',                       
'annual_inc_joint',               
'dti_joint',                      
'verification_status_joint',
'inq_fi',
'total_cu_tl',
'inq_last_12m'
]

to_drop = ['id', 'member_id', 'url', 'desc', 'pymnt_plan', 'policy_code'] + big_miss




data = data.drop(to_drop, axis=1)




data = data.drop('loan_status', axis=1)




to_do = ['emp_title', 'title']




data = data.drop(to_do, axis=1)




data.info()




null_d = data.isnull().sum(axis=0)
null_d[null_d>0]/data.shape[0]




cols = ['mths_since_last_delinq',           
        'mths_since_last_record',
        'mths_since_last_major_derog',
        'next_pymnt_d',
        'tot_coll_amt',                     
        'tot_cur_bal',      
        'total_rev_hi_lim']


for col in cols:
    display(data.groupby(data[col].isna())['ok_loan'].mean())
    
    
fig, axes=plt.subplots(3, 1, figsize=(10, 6))
for ax, col in zip(axes, ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']):
    data.groupby(data.ok_loan)[col].hist(bins=100, alpha=0.3, ax=ax)




data['issue_y'] = pd.to_datetime(data.issue_d).dt.year
data['issue_m'] = pd.to_datetime(data.issue_d).dt.month
data.drop(['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d'], inplace=True, axis=1)




null_d = data.isnull().sum(axis=0)
null_d[null_d>0]/data.shape[0]




mean_vals = data.select_dtypes('number').mean(axis=0)
data[mean_vals.index.values] = data[mean_vals.index.values].fillna(mean_vals)




data['has_last_pymnt_d'] = data.next_pymnt_d.isna()
data.drop(['city', 'next_pymnt_d', 'sub_grade'], axis=1, inplace=True)




null_d = data.isnull().sum(axis=0)
null_d[null_d>0]/data.shape[0]




from sklearn.preprocessing import LabelEncoder
cols = data.select_dtypes('object').columns
for col in cols:
    data[col] = LabelEncoder().fit_transform(data[col])




data.info()




from sklearn.model_selection import train_test_split
y = data.ok_loan
X = data.drop('ok_loan', axis=1)

X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3)




from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression




from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix, accuracy_score




def performance_summary(y_true, y_pred, y_score=None):
    # compute all the metrics
    numerics = {}
    numerics['auc'] = roc_auc_score(y_true, y_pred)
    numerics['precision'], numerics['recall'], numerics['f1'], _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    numerics['acc'] = accuracy_score(y_true, y_pred)
    
    basic = pd.DataFrame(numerics, index=[0])
    cm = confusion_matrix(y_true, y_pred)
    
    
    # vis
    _, axes = plt.subplots(1, 3, figsize=(12, 3))

    # 1. basic
    basic.T.plot.barh(xlim=(0, 1), legend=False, ax=axes[0], color=sns.color_palette(palette='plasma'), alpha=0.6)
    for p in axes[0].patches:
        axes[0].annotate('%.4f'%p.get_width(), (p.get_width()-0.2, p.get_y()+0.2), color='w')
    axes[0].set_title('Basic')   
        
    # 2.confusion matrix
    sns.heatmap(cm, annot=cm/cm.sum(), ax=axes[1], cmap="plasma", square=True, alpha=0.6)
    axes[1].set_title('Confusion Matrix')
    
    
    # 3. roc
    try:
        rc = roc_curve(y_true, y_score, pos_label=1)
        axes[2].plot(rc[0], rc[1])
        axes[2].fill_between(rc[0], rc[1], 0, alpha=0.3, color=sns.color_palette(palette='plasma'))
        axes[2].set_title('AUC = %.2f'%numerics['auc'])
    except:
        return
    




gbc = GradientBoostingClassifier().fit(X_tr, y_tr)

y_p = gbc.predict(X_t)
y_s = gbc.decision_function(X_t)




performance_summary(y_t, y_p, y_s)




impt = pd.Series(gbc.feature_importances_, index=X_tr.columns)
impt[impt>0].sort_values(ascending=False)




lgbc = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=20, n_jobs=2)




lgbc.fit(X_tr, y_tr)
y_p = lgbc.predict(X_t)




performance_summary(y_t, y_p)




impt = pd.Series(lgbc.feature_importances_, index=X_tr.columns)
impt.sort_values(ascending=False)




reduced_cols = impt[impt>0].index




X_tr_rd = X_tr[reduced_cols]
X_t_rd = X_t[reduced_cols]

X_tr_rd = X_tr_rd.drop('total_rec_prncp', axis=1)
X_t_rd = X_t_rd.drop('total_rec_prncp', axis=1)




gbc = GradientBoostingClassifier().fit(X_tr_rd, y_tr)

y_p = gbc.predict(X_t_rd)
y_s = gbc.decision_function(X_t_rd)




performance_summary(y_t, y_p, y_s)




lr = LogisticRegression()
lr.fit(X_tr, y_tr)
y_p = gbc.predict(X_t)
y_s = gbc.decision_function(X_t)
performance_summary(y_t, y_p, y_s)




impt = pd.Series(lr.coef_[0], index=X_tr.columns)
impt.sort_values()




for i in range(5, 0, -1):
    mini_cols = impt.abs().sort_values(ascending=False).head(i).index
    X_tr_mini = X_tr[mini_cols]
    X_t_mini = X_t[mini_cols]


    lr.fit(X_tr_mini, y_tr)
    y_p = lr.predict(X_t_mini)
    y_s = lr.decision_function(X_t_mini)

    performance_summary(y_t, y_p, y_s)






