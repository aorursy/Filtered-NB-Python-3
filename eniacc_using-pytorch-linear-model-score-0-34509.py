#!/usr/bin/env python
# coding: utf-8



# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split # 交叉验证 与将数据随机的分为训练集与数据集
                                                                      # 交叉验证 一般被用于评估一个机器学习模型的表现 https://zhuanlan.zhihu.com/p/32627500
from sklearn.preprocessing import StandardScaler #归一化数据  通过去除均值并缩放到单位方差来标准化特征
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
     # RidgeCV 带有内置交叉验证的岭回归。
from sklearn.metrics import mean_squared_error, make_scorer
     # mean_squared_error   均方误差损失函数
from scipy.stats import skew #函数计算数据集的偏度
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')
#njobs = 4




import os
print(os.listdir("../input"))




# Get data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("train : " + str(train.shape))




# Check for duplicates
# 检查重复项
idsUnique = len(set(train.Id))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " 重复 IDs for " + str(idsTotal) + " total entries")

# Drop Id column
train.drop("Id", axis = 1, inplace = True)




# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
# 寻找异常值
# matplotlib作可视化
plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

train = train[train.GrLivArea < 4000] #去除超过4000平的房子




# Log transform the target for official scoring
# 对偏度比较大的数据用log1p函数进行转化，使其更加服从高斯分布，此步处理可能会使我们后续的分类结果得到一个更好的结果；
train.SalePrice = np.log1p(train.SalePrice)
y = train.SalePrice
all_feature = pd.concat((train.iloc[:, :-1], test.iloc[:, 1:]))




all_feature




# Handle missing values for features where median/mean or most common value doesn't make sense
# 处理那些使用中位值和平均值填充没有意义的缺失值

# Alley : data description says NA means "no alley access"
all_feature.loc[:, "Alley"] = all_feature.loc[:, "Alley"].fillna("None") # NA 替换成 None
# BedroomAbvGr : NA = 0
all_feature.loc[:, "BedroomAbvGr"] = all_feature.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
# BsmtQual 地下室高度
all_feature.loc[:, "BsmtQual"] = all_feature.loc[:, "BsmtQual"].fillna("No")
# BsmtCond 地下室的一般情况
all_feature.loc[:, "BsmtCond"] = all_feature.loc[:, "BsmtCond"].fillna("No")
# BsmtExposure 地下室墙
all_feature.loc[:, "BsmtExposure"] = all_feature.loc[:, "BsmtExposure"].fillna("No")
# BsmtFinType1 地下室质量
all_feature.loc[:, "BsmtFinType1"] = all_feature.loc[:, "BsmtFinType1"].fillna("No")
# BsmtFinType2 第二加工区的质量（如果存在）
all_feature.loc[:, "BsmtFinType2"] = all_feature.loc[:, "BsmtFinType2"].fillna("No")
# BsmtFullBath 地下室全浴室
all_feature.loc[:, "BsmtFullBath"] = all_feature.loc[:, "BsmtFullBath"].fillna(0)
# BsmtHalfBath 地下室半浴室
all_feature.loc[:, "BsmtHalfBath"] = all_feature.loc[:, "BsmtHalfBath"].fillna(0)
# BsmtUnfSF 地下室未完成的平方英尺
all_feature.loc[:, "BsmtUnfSF"] = all_feature.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No 中央空调
all_feature.loc[:, "CentralAir"] = all_feature.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
# 靠近主要公路或铁路
all_feature.loc[:, "Condition1"] = all_feature.loc[:, "Condition1"].fillna("Norm")
all_feature.loc[:, "Condition2"] = all_feature.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
# 封闭的门廊面积（平方英尺）
all_feature.loc[:, "EnclosedPorch"] = all_feature.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
# ExterQual：外部材料质量
# ExterCond：外部材料的当前状态
all_feature.loc[:, "ExterCond"] = all_feature.loc[:, "ExterCond"].fillna("TA")
all_feature.loc[:, "ExterQual"] = all_feature.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
# 围栏质量
all_feature.loc[:, "Fence"] = all_feature.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
# 壁炉数量 FireplaceQu：壁炉质量
all_feature.loc[:, "FireplaceQu"] = all_feature.loc[:, "FireplaceQu"].fillna("No")
all_feature.loc[:, "Fireplaces"] = all_feature.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
# 家庭功能等级
all_feature.loc[:, "Functional"] = all_feature.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
# 车库位置
all_feature.loc[:, "GarageType"] = all_feature.loc[:, "GarageType"].fillna("No")
# 车库内部装修
all_feature.loc[:, "GarageFinish"] = all_feature.loc[:, "GarageFinish"].fillna("No")
# 车库质量
all_feature.loc[:, "GarageQual"] = all_feature.loc[:, "GarageQual"].fillna("No")
# 车库条件
all_feature.loc[:, "GarageCond"] = all_feature.loc[:, "GarageCond"].fillna("No")
all_feature.loc[:, "GarageArea"] = all_feature.loc[:, "GarageArea"].fillna(0)
# Size of garage in car capacity
all_feature.loc[:, "GarageCars"] = all_feature.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
all_feature.loc[:, "HalfBath"] = all_feature.loc[:, "HalfBath"].fillna(0)

# HeatingQC : NA most likely means typical
# 加热质量和条件
all_feature.loc[:, "HeatingQC"] = all_feature.loc[:, "HeatingQC"].fillna("TA")

# KitchenAbvGr : NA most likely means 0
# 客房总数（不包括浴室）
all_feature.loc[:, "KitchenAbvGr"] = all_feature.loc[:, "KitchenAbvGr"].fillna(0)

# KitchenQual : NA most likely means typical
# Kitchen quality
all_feature.loc[:, "KitchenQual"] = all_feature.loc[:, "KitchenQual"].fillna("TA")

# LotFrontage : NA most likely means no lot frontage
# 连接到物业的街道的线性英尺
all_feature.loc[:, "LotFrontage"] = all_feature.loc[:, "LotFrontage"].fillna(0)

# LotShape : NA most likely means regular
# 财产的总体形态
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")

# MasVnrType : NA most likely means no veneer
# 砖石饰面类型
# 砖石饰面面积（平方英尺）
all_feature.loc[:, "MasVnrType"] = all_feature.loc[:, "MasVnrType"].fillna("None")
all_feature.loc[:, "MasVnrArea"] = all_feature.loc[:, "MasVnrArea"].fillna(0)

# MiscFeature : data description says NA means "no misc feature"
# 其他类别未涵盖的杂项功能
# $其他功能的价值
all_feature.loc[:, "MiscFeature"] = all_feature.loc[:, "MiscFeature"].fillna("No")
all_feature.loc[:, "MiscVal"] = all_feature.loc[:, "MiscVal"].fillna(0)

# OpenPorchSF : NA most likely means no open porch
# 开放式阳台面积（平方英尺）
all_feature.loc[:, "OpenPorchSF"] = all_feature.loc[:, "OpenPorchSF"].fillna(0)

# PavedDrive : NA most likely means not paved
# 铺好的车道
all_feature.loc[:, "PavedDrive"] = all_feature.loc[:, "PavedDrive"].fillna("N")

# PoolQC : data description says NA means "no pool"
# 泳池品质
# 泳池大小
all_feature.loc[:, "PoolQC"] = all_feature.loc[:, "PoolQC"].fillna("No")
all_feature.loc[:, "PoolArea"] = all_feature.loc[:, "PoolArea"].fillna(0)

# SaleCondition : NA most likely means normal sale
# 销售条件
all_feature.loc[:, "SaleCondition"] = all_feature.loc[:, "SaleCondition"].fillna("Normal")

# ScreenPorch : NA most likely means no screen porch
# 屏幕门廊面积（以平方英尺为单位）
all_feature.loc[:, "ScreenPorch"] = all_feature.loc[:, "ScreenPorch"].fillna(0)

# TotRmsAbvGrd : NA most likely means 0
# 客房总数（不包括浴室）
all_feature.loc[:, "TotRmsAbvGrd"] = all_feature.loc[:, "TotRmsAbvGrd"].fillna(0)

# Utilities : NA most likely means all public utilities
# 可用的实用程序类型
all_feature.loc[:, "Utilities"] = all_feature.loc[:, "Utilities"].fillna("AllPub")

# WoodDeckSF : NA most likely means no wood deck
# 木制甲板面积（平方英尺）
all_feature.loc[:, "WoodDeckSF"] = all_feature.loc[:, "WoodDeckSF"].fillna(0)




# Some numerical features are actually really categories
all_feature = all_feature.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })




# Encode some categorical features as ordered numbers when there is information in the order
all_feature = all_feature.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )




# Create new features
# 1* Simplifications of existing features
all_feature["SimplOverallQual"] = all_feature.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
all_feature["SimplOverallCond"] = all_feature.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
all_feature["SimplPoolQC"] = all_feature.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
all_feature["SimplGarageCond"] = all_feature.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
all_feature["SimplGarageQual"] = all_feature.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
all_feature["SimplFireplaceQu"] = all_feature.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
all_feature["SimplFireplaceQu"] = all_feature.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
all_feature["SimplFunctional"] = all_feature.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
all_feature["SimplKitchenQual"] = all_feature.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
all_feature["SimplHeatingQC"] = all_feature.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
all_feature["SimplBsmtFinType1"] = all_feature.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
all_feature["SimplBsmtFinType2"] = all_feature.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
all_feature["SimplBsmtCond"] = all_feature.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
all_feature["SimplBsmtQual"] = all_feature.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
all_feature["SimplExterCond"] = all_feature.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
all_feature["SimplExterQual"] = all_feature.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

# 2* Combinations of existing features
# Overall quality of the house
all_feature["OverallGrade"] = all_feature["OverallQual"] * all_feature["OverallCond"]
# Overall quality of the garage
all_feature["GarageGrade"] = all_feature["GarageQual"] * all_feature["GarageCond"]
# Overall quality of the exterior
all_feature["ExterGrade"] = all_feature["ExterQual"] * all_feature["ExterCond"]
# Overall kitchen score
all_feature["KitchenScore"] = all_feature["KitchenAbvGr"] * all_feature["KitchenQual"]
# Overall fireplace score
all_feature["FireplaceScore"] = all_feature["Fireplaces"] * all_feature["FireplaceQu"]
# Overall garage score
all_feature["GarageScore"] = all_feature["GarageArea"] * all_feature["GarageQual"]
# Overall pool score
all_feature["PoolScore"] = all_feature["PoolArea"] * all_feature["PoolQC"]
# Simplified overall quality of the house
all_feature["SimplOverallGrade"] = all_feature["SimplOverallQual"] * all_feature["SimplOverallCond"]
# Simplified overall quality of the exterior
all_feature["SimplExterGrade"] = all_feature["SimplExterQual"] * all_feature["SimplExterCond"]
# Simplified overall pool score
all_feature["SimplPoolScore"] = all_feature["PoolArea"] * all_feature["SimplPoolQC"]
# Simplified overall garage score
all_feature["SimplGarageScore"] = all_feature["GarageArea"] * all_feature["SimplGarageQual"]
# Simplified overall fireplace score
all_feature["SimplFireplaceScore"] = all_feature["Fireplaces"] * all_feature["SimplFireplaceQu"]
# Simplified overall kitchen score
all_feature["SimplKitchenScore"] = all_feature["KitchenAbvGr"] * all_feature["SimplKitchenQual"]
# Total number of bathrooms
all_feature["TotalBath"] = all_feature["BsmtFullBath"] + (0.5 * all_feature["BsmtHalfBath"]) + all_feature["FullBath"] + (0.5 * all_feature["HalfBath"])
# Total SF for house (incl. basement)
all_feature["AllSF"] = all_feature["GrLivArea"] + all_feature["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
all_feature["AllFlrsSF"] = all_feature["1stFlrSF"] + all_feature["2ndFlrSF"]
# Total SF for porch
all_feature["AllPorchSF"] = all_feature["OpenPorchSF"] + all_feature["EnclosedPorch"] + all_feature["3SsnPorch"] + all_feature["ScreenPorch"]
# Has masonry veneer or not
all_feature["HasMasVnr"] = all_feature.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
# House completed before sale or not
all_feature["BoughtOffPlan"] = all_feature.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})




# Find most important features relative to target
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)




# Create new features
# 3* Polynomials on the top 10 existing features
all_feature["OverallQual-s2"] = all_feature["OverallQual"] ** 2
all_feature["OverallQual-s3"] = all_feature["OverallQual"] ** 3
all_feature["OverallQual-Sq"] = np.sqrt(all_feature["OverallQual"])
all_feature["AllSF-2"] = all_feature["AllSF"] ** 2
all_feature["AllSF-3"] = all_feature["AllSF"] ** 3
all_feature["AllSF-Sq"] = np.sqrt(all_feature["AllSF"])
all_feature["AllFlrsSF-2"] = all_feature["AllFlrsSF"] ** 2
all_feature["AllFlrsSF-3"] = all_feature["AllFlrsSF"] ** 3
all_feature["AllFlrsSF-Sq"] = np.sqrt(all_feature["AllFlrsSF"])
all_feature["GrLivArea-2"] = all_feature["GrLivArea"] ** 2
all_feature["GrLivArea-3"] = all_feature["GrLivArea"] ** 3
all_feature["GrLivArea-Sq"] = np.sqrt(all_feature["GrLivArea"])
all_feature["SimplOverallQual-s2"] = all_feature["SimplOverallQual"] ** 2
all_feature["SimplOverallQual-s3"] = all_feature["SimplOverallQual"] ** 3
all_feature["SimplOverallQual-Sq"] = np.sqrt(all_feature["SimplOverallQual"])
all_feature["ExterQual-2"] = all_feature["ExterQual"] ** 2
all_feature["ExterQual-3"] = all_feature["ExterQual"] ** 3
all_feature["ExterQual-Sq"] = np.sqrt(all_feature["ExterQual"])
all_feature["GarageCars-2"] = all_feature["GarageCars"] ** 2
all_feature["GarageCars-3"] = all_feature["GarageCars"] ** 3
all_feature["GarageCars-Sq"] = np.sqrt(all_feature["GarageCars"])
all_feature["TotalBath-2"] = all_feature["TotalBath"] ** 2
all_feature["TotalBath-3"] = all_feature["TotalBath"] ** 3
all_feature["TotalBath-Sq"] = np.sqrt(all_feature["TotalBath"])
all_feature["KitchenQual-2"] = all_feature["KitchenQual"] ** 2
all_feature["KitchenQual-3"] = all_feature["KitchenQual"] ** 3
all_feature["KitchenQual-Sq"] = np.sqrt(all_feature["KitchenQual"])
all_feature["GarageScore-2"] = all_feature["GarageScore"] ** 2
all_feature["GarageScore-3"] = all_feature["GarageScore"] ** 3
all_feature["GarageScore-Sq"] = np.sqrt(all_feature["GarageScore"])




# Differentiate numerical features (minus the target) and categorical features
categorical_features = all_feature.select_dtypes(include = ["object"]).columns
numerical_features = all_feature.select_dtypes(exclude = ["object"]).columns
# numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = all_feature[numerical_features]
train_cat = all_feature[categorical_features]




# Handle remaining missing values for numerical features by using median as replacement
# 处理数字特征缺失值
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))




# 对偏的数值特征进行对数变换以减少离群值的影响
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
# 根据一般经验，绝对值大于0.5的偏斜至少被认为是中等偏斜
skewness = train_num.apply(lambda x: skew(x)) # 计算数字特征的偏斜
skewness = skewness[abs(skewness) > 0.5] #取绝对值大于0.5的偏斜数据
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index #取绝对值大于0.5的偏斜数据index
train_num[skewed_features] = np.log1p(train_num[skewed_features]) #对绝对值大于0.5的偏斜的数据进行log1p变幻




# Create dummy features for categorical values via one-hot encoding
# 分类特征转化为one-hot向量
print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat) #one-hot编码
print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))




# Join categorical and numerical features
all_feature = pd.concat([train_num, train_cat], axis = 1) #按列合并数字特征和分类特征

# Standardize numerical features
# 标准化数字特征
stdSc = StandardScaler()
all_feature.loc[:, numerical_features] = stdSc.fit_transform(all_feature.loc[:, numerical_features])




n_train = train.shape[0]
train = all_feature[:n_train]
test_feature = all_feature[n_train:]
print("New number of features : " + str(train.shape[1]))




# Partition the dataset in train + validation sets
# 分离数据集与验证集
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))




X_train.isnull().values.sum()




import torch
from torch import nn
from torch.nn import init




num_inputs, num_outputs, num_hiddens_1,num_hiddens_2,num_hiddens_3,num_hiddens_4 = 324, 1, 256,170,85,18
lr=0.5
decay=1e-3
net = nn.Sequential(
        nn.Linear(num_inputs, num_hiddens_1),# 隐藏层
        nn.Tanh(),                         # 激活函数
#        nn.ReLU(),
#       nn.Sigmoid(),
        nn.Linear(num_hiddens_1, num_hiddens_2),
        nn.Tanh(),
#       nn.ReLU(),
#       nn.Sigmoid(),
        nn.Linear(num_hiddens_2, num_hiddens_3),
        nn.Tanh(),
#       nn.ReLU(),
#       nn.Sigmoid(),    
        nn.Linear(num_hiddens_3, num_hiddens_4),
        nn.Tanh(),
        nn.Linear(num_hiddens_4, num_outputs), # 输出层
        )




for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
batch_size = 85

X_tr = X_train.values
X_te = X_test.values
y_tr = y_train.values
y_te = y_test.values
X_tr_torch = torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float).view(-1,324),torch.tensor(y_tr, dtype=torch.float).view(-1,1))
X_te_torch = torch.utils.data.TensorDataset(torch.tensor(X_te, dtype=torch.float).view(-1,324),torch.tensor(y_te, dtype=torch.float).view(-1,1))
# pytorch 读取pandas的方法
train_iter = torch.utils.data.DataLoader(X_tr_torch, batch_size=batch_size, shuffle=True )
test_iter = torch.utils.data.DataLoader(X_te_torch, batch_size=batch_size, shuffle=False)




loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=decay)




def train_net(net, train_iter,loss, num_epochs, batch_size, lr, optimizer):
    for epochs in range(num_epochs):
        for x,y in train_iter:
            y_hat=net(x) #模型计算
            l=loss(y_hat,y).sum() #损失计算
            optimizer.zero_grad() #梯度清0
            l.backward() #反向传播
            optimizer.step()
        print('epochs:'+str(epochs)+'loss:'+str(l))




lr=0.001
num_epochs = 500

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
    
train_net(net, train_iter,loss, num_epochs, batch_size, lr, optimizer)




torch.cuda.is_available()




def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()




log_rmse(net,torch.tensor(X_te, dtype=torch.float).view(-1,324),torch.tensor(y_te, dtype=torch.float).view(-1,1))




print(os.listdir("../"))




test_feature = torch.tensor(test_feature.values, dtype=torch.float)




preds = net(test_feature).detach().numpy()




test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
submission.SalePrice=np.expm1(submission.SalePrice)




submission.to_csv('submission.csv', index=False)




submission

