#!/usr/bin/env python
# coding: utf-8



ntrain=train.shape[0]
ntest=test.shape[0]
SEED=0
NFOLDS=5
kf=KFold(ntrain,n_folds=NFOLDS,random_state=SEED)




class Sk_learnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state']=seed
        self.clf=clf(**params)
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
        
    def predict(self,x):
        return self.clf.predict(x)
    def fit(self,x,y):
        return self.clf.fit(x,y)
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_) 
     




def get_oof(clf,x_train,y_train,x_test):
    oof_train=np.zeros((ntrain,))
    oof_test=np.zeros((ntest,))
    oof_test_skf=np.empty((NFOLDS,ntest))
    for i,(train_index,test_index) in enumerate(kf):
        x_tr=x_train[train_index]
        y_tr=y_train[train_index]
        x_te=x_train[test_index]
        
        clf.train[x_tr,y_tr]
        
        oof_train[test_index]=clf.predict(x_te)
        oof_test_skf[i,:]=clf.predict(x_test)
        
    oof_test[:]=oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
    




#randome forest parameters
rf_params={
    'n_jobs'=-1
    'n_estimators':500,
    'warm_start':True,
    'max_depth':6,
    'min_sampels_leaf':2,
    'max_features':'sqrt',
    'verbose':0
}
#Extra Trees Parameters
et_params={
    'n_jobs':-1,
    'n_estimators':500,
    'warm_start':True,
    'max_depth':6,
    'min_samples_leaf':2,
    'verbose':0
}
#AdaBoost parameter
ada_params={
    'n_estimators':500,
    'learning_rate':0.75
}
#Gradient Boosting parameters
gb_params={
    'n_estimators':500,
    'max_detph':5,
    'min_samples_leaf':2,
    'verbose':0
}
#Support Vector Classifier parameter
svc_params={
    'kernel':'linear',
    'C':0.025
}




# rf=SklearnHelper(clf=RandomForestClassifier,seed=SEED,params=rf_params)
# et=SklearnHelper(clf=ExtraTreesClassifier,seed=SEED,params=et_params)
# ada=SklearnHelper(clf=AdaBoostClassifier,seed=SEED,params=ada_params)
# gb=SklearnHelper(clf=GradientBoostingClassifier,seed=SEED,params=gb_params)
# svc=SklearnHelper(clf=SVC,seed=SEED,params=svc_params)




# y_train=train['Survived'].ravel()
# train=train.drop(['Survived'],axis=1)
# x_train=train.values
# x_test=test.values




# et_oof_train,et_oof_test=get_oof(et,x_train,y_train,x_test)
# xf_oof_train,rf_oof_test=get_oof(rf,x_train,y_train,x_test)
# ada_oof_train,ada_oof_test=get_oof(ada,x_train,y_trin,x_test)
# gb_oof_train,gb_oof_test=get_oof(gb,x_train,y_train,x_test)
# svc_oof_train,svc_oof_test=get_oof(svc,x_train,y_train,x_test)




# rf_featrue=rf.feature_importance(x_train,y_train)
# et_feature=et.feature_importance(x_train,y_train)
# ada_feature=ada.feature_importance(x_train,y_trian)
# gb_feature=gb.feature_importance(x_train,y_train)




# cols=train.columns.values
# feature_dataframe=pd.DataFrame({'features':cols,
#                                'Random Forest feature importances':rf_features,
#                                'Extra Tress feature importances':et_features,
#                                'AdaBoost feature importances':ada_features,
#                                'Gradient Boost feature importances':gb_features})




# gbm=xgb.XGBClassifier(
#     n_estimators=2000,
#     max_depth=4,
#     min_child_weght=2,
#     gamma=0.9,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=-1,
#     scale_pos_weight=1).fit(x_trian,y_train)
# predictions=gbm.predict(x_test)




# StackingSubmission=pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})
# StackingSubmission.to_csv("StackingSubmission.csv",index=False)






