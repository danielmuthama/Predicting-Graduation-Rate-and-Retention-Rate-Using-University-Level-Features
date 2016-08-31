# Import libraries necessary for this project
import operator
import numpy as np
from scipy import stats
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt

# Load The Counted 2015 dataset
data = pd.read_csv("CollegeScorecard_Raw_Data/MERGED2013_PP.csv")
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
###########Processing the dataset to create the finalized dataset############
data = data[(data.main==1)]
#From 7804 only 5709 left. Universities like Kaplan or Remington that have more than one branch
#arent represented multiple times.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.PREDDEG>=1)]
data = data[(data.PREDDEG<=3)]
#From 5709 only 5409 left. Institutions whose predominant degree awarded is not classified or
#entirely graduate-degree granting are not represented.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.HIGHDEG>=1)]
#From 5409 only 5409 left. Institutions who do not grant degrees are not represented.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.region>=1)]
#From 5409 only 5408 left. US Service Schools are not represented.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.LOCALE <= 43)]
#From 5408 only 5405 left. No location environment for some reason. Location like rural, etc.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.ADM_RATE <= 1)]
#From 5405 only 1877 left. Remove ones with no ADM_RATE.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.SAT_AVG <= 1600)]
#From 1877, only 1371 left. The ones that did not have an SAT_AVG were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.CURROPER == 1)]
#From 1371 only 1366 left. The ones that were not currently operating were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.COSTT4_A <= 10000000)]
#From 1366 only 1360 left. The ones that did not have an average cost of attendance were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.PFTFAC<=1)]
#From 1360 only 1350 left. The ones that did not have an proportion of faculty
#that is full-time were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['GRAD_RATE'] = pd.Series(data.C150_4, index=data.index)
data['GRAD_RATE'].fillna(data['C150_L4'], inplace=True)
data = data[(data.GRAD_RATE<=1)]
#From 1350 only 1344 left. Using the graduation rates at 4 year institutions(C150_4) and the
#graduation rates from less than 4 year institutions(C150_L4), I made a variable that combines them
#together(GRAD_RATE). Then I remove the observations with no grad rates from either.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['RET_RATE'] = pd.Series(data.RET_FT4, index=data.index)
data['RET_RATE'].fillna(data['RET_FTL4'], inplace=True)
data = data[(data.RET_RATE<=1)]
#From 1344 only 1340 left. Using the full-time retention rates at 4 year institutions(RET_FT4) and the
#retention rates from less than 4 year institutions(RET_FTL4), I made a variable that combines them
#together(RET_RATE). Then I remove the observations with no retention rates from either.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[(data.UG25abv <= 1)]
#From 1340 only 1335 left. The ones that did not have a percentage of undergraduates above 25
#were removed
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['PAR_ED_PCT_1STGEN'] = pd.to_numeric(data['PAR_ED_PCT_1STGEN'],errors=10)
data = data[(data.PAR_ED_PCT_1STGEN<=1)]
#From 1335 only 1306 left. The ones that did not have a number for percent of 1st generation
#students were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['PAR_ED_PCT_MS'] = pd.to_numeric(data['PAR_ED_PCT_MS'],errors=10)
data = data[(data.PAR_ED_PCT_MS<=1)]
#From 1306 only 1287 left. The ones that did not have a number for percent of students whose parents'
#highest level of education is middle school were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['PAR_ED_PCT_HS'] = pd.to_numeric(data['PAR_ED_PCT_HS'],errors=10)
data = data[(data.PAR_ED_PCT_HS<=1)]
#From 1287 only 1287 left. The ones that did not have a number for percent of students whose parents'
#highest level of education is high school were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data['PAR_ED_PCT_PS'] = pd.to_numeric(data['PAR_ED_PCT_PS'],errors=10)
data = data[(data.PAR_ED_PCT_PS<=1)]
#From 1287 only 1287 left. The ones that did not have a number for percent of students whose parents'
#highest level of education is post-secondary were removed.
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data.reset_index(drop=True)
all_outliers = np.array([], dtype='int64')
GQ1 = np.percentile(data['GRAD_RATE'], 25)
GQ3 = np.percentile(data['GRAD_RATE'], 75)
Gstep = 1.5*(GQ3-GQ1)
Goutlier_points = data[~((data['GRAD_RATE'] >= GQ1 - Gstep) & (data['GRAD_RATE'] <= GQ3 + Gstep))]
all_outliers = np.append(all_outliers, Goutlier_points.index.values.astype('int64'))
RQ1 = np.percentile(data['RET_RATE'], 25)
RQ3 = np.percentile(data['RET_RATE'], 75)
Rstep = 1.5*(RQ3-RQ1)
Routlier_points = data[~((data['RET_RATE'] >= RQ1 - Rstep) & (data['RET_RATE'] <= RQ3 + Rstep))]
all_outliers = np.append(all_outliers, Routlier_points.index.values.astype('int64'))
outliers  = []
all_outlier, indices = np.unique(all_outliers, return_inverse=True)
counts = np.bincount(indices)
outliers = all_outlier[counts>=1]
print outliers
data = data.drop(data.index[outliers]).reset_index(drop = True)
#Found the outliers for GRAD_RATE and RET_RATE and removed them. 
print "The 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
data = data[['PREDDEG','HIGHDEG','CONTROL','region','LOCALE','ADM_RATE','SAT_AVG','COSTT4_A',
             'PFTFAC','GRAD_RATE','RET_RATE','UG25abv','PAR_ED_PCT_1STGEN','PAR_ED_PCT_MS',
             'PAR_ED_PCT_HS','PAR_ED_PCT_PS']]
print "The finalized 2013 Raw_Data dataset has {} samples with {} features each.".format(*data.shape)
##################################################
#############Preliminary Statistics on Finalized Dataset########
display(data[['GRAD_RATE','RET_RATE']].describe())
display(data[['ADM_RATE','SAT_AVG','COSTT4_A']].describe())
print stats.mstats.mode(data[['PREDDEG','HIGHDEG','CONTROL','region','LOCALE']])
###################################################
##############Exploratory Visualizations##############
#Hexbin of Graduation rate versus Retention rate with linear regresion line, R, and R^2 values
data.plot.hexbin(x='GRAD_RATE',y='RET_RATE',gridsize=35)
slope, intercept, r_value, p_value, std_err = stats.linregress(data['GRAD_RATE'],data['RET_RATE'])
r_squared = r_value**2
r_squared = r_squared.round(decimals=3)
plt.plot(data['GRAD_RATE'],slope*data['GRAD_RATE']+intercept,'-',color='Red')
plt.show()
#Scatter plot of Admission rate versus SAT average with Graduation rate filling the circles
data.plot.scatter(x='ADM_RATE',y='SAT_AVG',c='GRAD_RATE',alpha=0.99);
plt.show()
#Scatter plot of Admission rate versus SAT average with Retention rate filling the circles
data.plot.scatter(x='ADM_RATE',y='SAT_AVG',c='RET_RATE',alpha=0.99);
plt.show()
#Box plot of graduation and retention rate
data[['GRAD_RATE','RET_RATE']].plot.box();
plt.show()
###################################################
#######Create needed functions################
def Kfold_DTR(length_of_data, nfolds, rs, feats, target):
    kf = KFold(n=length_of_data, n_folds=nfolds, shuffle=True, random_state=rs)
    regressor = DecisionTreeRegressor(random_state=rs)
    scores = np.zeros((nfolds,1))
    for k, (train, test) in enumerate(kf):
        X_train = feats.drop(feats.index[test]).reset_index(drop = True)
        y_train = target.drop(target.index[test]).reset_index(drop = True)
        X_test = feats.drop(feats.index[train]).reset_index(drop = True)
        y_test = target.drop(target.index[train]).reset_index(drop = True)
        regressor.fit(X_train, y_train)
        scores[k,] = regressor.score(X_test,y_test)
    print "{} folds, Decision Tree Regressor r^2: {}".format(nfolds, scores)
    print 'Average:', np.mean(scores)
def PCAs(ncomponents, feats):
    pca = PCA(n_components=ncomponents, whiten=True)
    pca.fit(feats)
    print (pca.explained_variance_ratio_)
    pca_results = rs.pca_results(feats, pca)
    plt.show()
def GSCV(rs, nfolds, feats, target):
    regressor = DecisionTreeRegressor(random_state=rs)
    r2_scorer = make_scorer(score_func=r2_score)
    parameters = {'max_depth':list(range(1,21)), 'min_samples_split':list(range(2,10)),
                  'min_samples_leaf':list(range(1,10))}
    grid_obj = GridSearchCV(estimator=regressor, cv=nfolds, param_grid=parameters, scoring=r2_scorer)
    grid_obj = grid_obj.fit(feats, target)
    pred = grid_obj.predict(feats)
    print "Tuned model grad has a r^2 score of {:.4f}.".format(r2_score(target, pred))
    print "Best params: {}".format( grid_obj.best_params_ )
def PCAev(ncomponents, feats):
    pca = PCA(n_components=10, whiten=True)
    pca.fit(features_st)
    print (pca.explained_variance_ratio_.cumsum())
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()
def Kfold_DTR_OptParms(length_of_data, nfolds, rs, md, mss, msl, feats, target, GoR):
    kf = KFold(n=length_of_data, n_folds=nfolds, shuffle=True, random_state=rs)
    regressor = DecisionTreeRegressor(max_depth=md, min_samples_split=mss,
                                      min_samples_leaf=msl, random_state=rs)
    scores = np.zeros((nfolds,1))
    for k, (train, test) in enumerate(kf):
        X_train = feats.drop(feats.index[test]).reset_index(drop = True)
        y_train = target.drop(target.index[test]).reset_index(drop = True)
        X_test = feats.drop(feats.index[train]).reset_index(drop = True)
        y_test = target.drop(target.index[train]).reset_index(drop = True)
        regressor.fit(X_train, y_train)
        scores[k,] = regressor.score(X_test,y_test)
    print "{} , random_state= {}".format(GoR, rs)
    print 'Average:', np.mean(scores)
    print 'Standard Deviation:', np.std(scores)
##################Methodology Implementation##################
#Separate features and individual target variables
features = data.drop(['GRAD_RATE','RET_RATE'],axis=1).reset_index(drop=True)
grad = data['GRAD_RATE'].reset_index(drop=True)
ret = data['RET_RATE'].reset_index(drop=True)
#Implement KFold Cross Validation technique and Decision Tree Regressor for Grad
#Also find the r^2 scores for each fold and the average
Kfold_DTR(len(data), 9, 42, features, grad)
#Implement KFold Cross Validation technique and Decision Tree Regressor for Ret
#Also find the r^2 scores for each fold and the average
Kfold_DTR(len(data), 9, 42, features, ret)
#Apply clustering algorithm of choice to the data, getting silhouette scores
#for multiple components
i=0
sil_scores = np.zeros((20,1))
while i <= 19:
    clusterer = GMM(n_components=i+2, covariance_type='full', random_state=42)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    sil_scores[i,] = silhouette_score(data, preds, random_state=42)
    i += 1
#Finding the maximum silhouette score and optimal number of clusters
max_index, max_value = max(enumerate(sil_scores), key=operator.itemgetter(1))
print 'Based on the highest silhouette score of', max_value, 'the number of clusters is', max_index+2
#Splitting the dataset based on the optimal number of clusters
clusterer = GMM(n_components=max_index+2, covariance_type='full', random_state=42)
clusterer.fit(data)
preds = clusterer.predict(data)
Cluster1 = data.drop(data.index[np.where(preds==1)[0]])
Cluster2 = data.drop(data.index[np.where(preds==0)[0]])
#Separate features and target variables for each cluster
#Cluster 1
C1_feat = Cluster1.drop(['GRAD_RATE','RET_RATE'],axis=1).reset_index(drop=True)
C1_grad = Cluster1['GRAD_RATE'].reset_index(drop=True)
C1_ret = Cluster1['RET_RATE'].reset_index(drop=True)
#Cluster 2
C2_feat = Cluster2.drop(['GRAD_RATE','RET_RATE'],axis=1).reset_index(drop=True)
C2_grad = Cluster2['GRAD_RATE'].reset_index(drop=True)
C2_ret = Cluster2['RET_RATE'].reset_index(drop=True)
#Do a PCA on each cluster's features separately
#Cluster 1
PCAs(6, C1_feat)
#Cluster 2
PCAs(6, C2_feat)
#Transform the data based on the one component explaining most of the variance
#Cluster 1
pca = PCA(n_components=1, whiten=True)
pca.fit(C1_feat)
C1_feat_tr = pca.transform(C1_feat)
#Cluster 2
pca = PCA(n_components=1, whiten=True)
pca.fit(C2_feat)
C2_feat_tr = pca.transform(C2_feat)
####Cluster 1####
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Graduation rates on Cluster 1's transformed data.
C1_feat_tr.shape
#Since Cluster 1 has 763 institutions, a 7 fold will be used.
#Using transformed data. Find the r^2 scores for each fold and the average
#C1_feat_tr must be made into a DataFrame
C1_feat_tr = pd.DataFrame(C1_feat_tr).reset_index(drop = True)
Kfold_DTR(len(C1_feat_tr), 7, 42, C1_feat_tr, C1_grad)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Retention rates on Cluster 1's transformed data.
C1_feat_tr.shape
#Since Cluster 1 has 763 institutions, a 7 fold will be used.
#Using transformed data. Find the r^2 scores for each fold and the average
#C1_feat_tr must be made into a DataFrame
C1_feat_tr = pd.DataFrame(C1_feat_tr).reset_index(drop = True)
Kfold_DTR(len(C1_feat_tr), 7, 42, C1_feat_tr, C1_ret)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Graduation rates on Cluster 1's data.
C1_feat.shape
#Since Cluster 1 has 763 institutions, a 7 fold will be used.
#Find the r^2 scores for each fold and the average
#Cluster1 must be made into a DataFrame
C1_feat = pd.DataFrame(C1_feat).reset_index(drop = True)
Kfold_DTR(len(C1_feat), 7, 42, C1_feat, C1_grad)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Retention rates on Cluster 1's data.
C1_feat.shape
#Since Cluster 1 has 763 institutions, a 7 fold will be used.
#Find the r^2 scores for each fold and the average
#Cluster1 must be made into a DataFrame
C1_feat = pd.DataFrame(C1_feat).reset_index(drop = True)
Kfold_DTR(len(C1_feat), 7, 42, C1_feat, C1_ret)
####Cluster 2####
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Graduation rates on Cluster 2's transformed data.
C1_feat_tr.shape
#Since Cluster 2 has 515 institutions, a 5 fold will be used.
#Using transformed data. Find the r^2 scores for each fold and the average
#C2_feat_tr must be made into a DataFrame
C2_feat_tr = pd.DataFrame(C2_feat_tr).reset_index(drop = True)
Kfold_DTR(len(C2_feat_tr), 5, 42, C2_feat_tr, C2_grad)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Retention rates on Cluster 2's transformed data.
C2_feat_tr.shape
#Since Cluster 2 has 515 institutions, a 5 fold will be used.
#Using transformed data. Find the r^2 scores for each fold and the average
#C2_feat_tr must be made into a DataFrame
C2_feat_tr = pd.DataFrame(C2_feat_tr).reset_index(drop = True)
Kfold_DTR(len(C2_feat_tr), 5, 42, C2_feat_tr, C2_ret)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Graduation rates on Cluster 2's data.
C2_feat.shape
#Since Cluster 2 has 515 institutions, a 5 fold will be used.
#Find the r^2 scores for each fold and the average
#Cluster2 must be made into a DataFrame
C2_feat = pd.DataFrame(C2_feat).reset_index(drop = True)
Kfold_DTR(len(C2_feat), 5, 42, C2_feat, C2_grad)
#Implement KFold Cross Validation technique and Decision Tree Regressor for 
#Retention rates on Cluster 2's data.
C2_feat.shape
#Since Cluster 2 has 515 institutions, a 5 fold will be used.
#Find the r^2 scores for each fold and the average
#Cluster2 must be made into a DataFrame
C2_feat = pd.DataFrame(C2_feat).reset_index(drop = True)
Kfold_DTR(len(C2_feat), 5, 42, C2_feat, C2_ret)
###################Refinement############################
#Perform a Grid Search on the data to see if parameters can be tuned to provide better
#results. Still fitting a Decision Tree Regressor and using a KFold cross validation
#with 9 folds. The scorer will be the highest r^2.
#For Graduation Rate
GSCV(42, 9, features, grad)
#For Retention Rate
GSCV(42, 9, features, ret)
#Standardize the features then perform a PCA. Observe the explained variance
#from each component and cut off once additional explained variance is not
#optimal to model. Transform features into desired components. Run the Grid Search
#Decision Tree Regressors for both target variables on the transformed data.
#Standardize the features
features_st = scale(features)
#Perform a PCA analysis and observe the explained variance
PCAev(10, features_st)
#Go with 8 components and transform the scaled features into these components
pca = PCA(n_components=8, whiten=True)
pca.fit(features_st)
feat_st_tr = pca.transform(features_st)
#Decision Tree regressor for transformed scaled features on graduation rate
GSCV(42, 9, feat_st_tr, grad)
#Decision Tree regressor for transformed scaled features on retention rate
GSCV(42, 9, feat_st_tr, ret)
#Perform a Grid Search on the clustered data to see if parameters can be tuned to provide
#better results. Still fitting a Decision Tree Regressor and using a KFold cross validation
#with 7 folds on the first cluster and 5 folds on the second cluster. The scorer will be
#the highest r^2.
#Cluster 1 - Graduation Rate
GSCV(42, 7, C1_feat, C1_grad)
#Cluster 1 - Retention Rate
GSCV(42, 7, C1_feat, C1_ret)
#Cluster 2 - Graduation Rate
GSCV(42, 5, C2_feat, C2_grad)
#Cluster 2 - Retention Rate
GSCV(42, 5, C2_feat, C2_ret)
#Standardize each institution's features on cluster 1 then perform a PCA. Observe the explained
#variance from each component and cut off once additional explained variance is not
#optimal to model. Transform cluster's features into desired components. Run the Grid Search
#Decision Tree Regressors for both target variables on cluster 1's transformed data.
#Standardize the features
C1_feat_st = scale(C1_feat)
#Perform a PCA analysis on cluster 1's standardized features and observe the explained variance
PCAev(10, C1_feat_st)
#Go with 7 components and transform the scaled features into these components
pca = PCA(n_components=7, whiten=True)
pca.fit(C1_feat_st)
C1_feat_st_tr = pca.transform(C1_feat_st)
#Decision Tree regressor for Cluster 1's transformed scaled features on graduation rate
GSCV(42, 7, C1_feat_st_tr, C1_grad)
#Decision Tree regressor for Cluster 1's transformed scaled features on retention rate
GSCV(42, 7, C1_feat_st_tr, C1_ret)
#Standardize each institution's features on cluster 2 then perform a PCA. Observe the explained
#variance from each component and cut off once additional explained variance is not
#optimal to model. Transform cluster's features into desired components. Run the Grid Search
#Decision Tree Regressors for both target variables on cluster 2's transformed data.
#Standardize the features
C2_feat_st = scale(C2_feat)
#Perform a PCA analysis on cluster 2's standardized features and observe the explained variance
PCAev(10, C2_feat_st)
#Go with 8 components and transform the scaled features into these components
pca = PCA(n_components=8, whiten=True)
pca.fit(C2_feat_st)
C2_feat_st_tr = pca.transform(C2_feat_st)
#Decision Tree regressor for Cluster 2's transformed scaled features on graduation rate
GSCV(42, 5, C2_feat_st_tr, C2_grad)
#Decision Tree regressor for Cluster 2's transformed scaled features on retention rate
GSCV(42, 5, C2_feat_st_tr, C2_ret)
###################Results#############
#Finding some proof of robustness in the final model
#Use the optimal parameters for the final model
#The Decision tree regressor onto the features, no clusters, no components
#Implement KFold Cross Validation technique and Decision Tree Regressor for Grad
#Also find the r^2 scores for each fold and the average,
#random state=42
Kfold_DTR_OptParms(len(data), 9, 42, 6, 2, 8, features, grad, 'Grad')
#random state=15
Kfold_DTR_OptParms(len(data), 9, 15, 6, 2, 8, features, grad, 'Grad')
#random state=97
Kfold_DTR_OptParms(len(data), 9, 97, 6, 2, 8, features, grad, 'Grad')
#Implement KFold Cross Validation technique and Decision Tree Regressor for Ret
#Also find the r^2 scores for each fold and the average,
#random state=42
Kfold_DTR_OptParms(len(data), 9, 42, 6, 2, 6, features, ret, 'Ret')
#random state=15
Kfold_DTR_OptParms(len(data), 9, 15, 6, 2, 6, features, ret, 'Ret')
#random state=97
Kfold_DTR_OptParms(len(data), 9, 97, 6, 2, 6, features, ret, 'Ret')
##############Conclusion################
#Free Form Visualization
#A scatter plot of the true and predicted graduation rates from a decision tree
#regressor based on the final model. The model will be trained on 8/9 of the
#data and tested on 1/9 of the data
#random state = 106
X_train, X_test, y_train, y_test = train_test_split(features, grad, test_size=0.111, random_state=106)
regressor = DecisionTreeRegressor(max_depth=6, min_samples_split=2,
                                  min_samples_leaf=8, random_state=106)
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
preds = regressor.predict(X_test)
plt.scatter(y_test, preds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,preds)
plt.plot(y_test,slope*y_test+intercept,'-',color='Blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Graduation Rate: Actual Versus Predicted')
plt.show()
#A scatter plot of the true and predicted retention rates from a decision tree
#regressor based on the final model. The model will be trained on 8/9 of the
#data and tested on 1/9 of the data
#random state = 106
X_train, X_test, y_train, y_test = train_test_split(features, ret, test_size=0.111, random_state=106)
regressor = DecisionTreeRegressor(max_depth=6, min_samples_split=2,
                                  min_samples_leaf=6, random_state=106)
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
preds = regressor.predict(X_test)
plt.scatter(y_test, preds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,preds)
plt.plot(y_test,slope*y_test+intercept,'-',color='Blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Retention Rate: Actual Versus Predicted')
plt.show()
