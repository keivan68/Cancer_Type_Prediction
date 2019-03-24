# ## Syapse ML Challenge
# ## Cancer Type
# Keivan ebrahimi

# ## Importing the packages, and reading the data

# In[1]:
import time
t = time.time()
import os  
import pandas as pd
import numpy as np
np.random.seed()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# on or off
vis = 'on'
feature_scaling = 'on'
outlier_detection = 'on'
svm_main = 'off'
tuning_svm = 'off'
knn_main = 'on'
cross_validation = 'off'
categorical_var = 'on' # off (will remove class_id) / on (will keep class_id with one-hot encoding) / catboost (will keep the main class_id for using in catboost model) 

# suppress Future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set working directory
path = os.path.expanduser('~/Dropbox/Internship, Job/Job/Syapse/')
#path = os.path.expanduser('~/Documents/Tensorflow/Untitled Folder/dataset/')
#path = os.path.expanduser('~/Desktop/dataset/')
os.chdir(path)  
# Read in the data
df = pd.read_table('patients.txt')

print(df.shape)
df.columns.tolist()

# work with a small sample of data for faster computation
df = df.sample(frac = 1, random_state = 1)  
print(df.head())
print(df.shape)
df.iloc[:,:].head()

# In[3]:


print(df.apply(lambda x: sum(x.isnull()), axis=0))

df.fillna(0, inplace=True)
print(df.apply(lambda x: sum(x.isnull()), axis=0))
# ## Correlation heatmap

# In[ ]:


if vis=='on':
    correlation = df.corr()
    plt.figure(figsize=(10, 10))  
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')


# ## Pairwise plots
# ### take a small sample as this is computationally expensive

if vis=='on':
    df_sample = df.sample(frac=1)
    pplot = sns.pairplot(df_sample, hue="cancer_type")

# ## removing errors

# In[ ]:


d1=df.sex.tolist()
d2=df.cancer_type.tolist()
remove_indices=[]
for i in range(df.shape[0]):
    if (d1[i]=='female' and d2[i]=='prostate') or (d1[i]=='male' and (d2[i]=='ovary' or d2[i]=='cervix')):
        #if i not in remove_indices:
        remove_indices.append(i)
    #if d1[i]!='male' and d1[i]!='female':
        #if i not in remove_indices:
        #remove_indices.append(i)
print(len(remove_indices))

d3=df.age.tolist()
d4=df['height(in)'].tolist()
for i in range(df.shape[0]):
    if  d3[i]<0 or (d4[i]<17 and d3[i]>1) or d4[i]<10:
        remove_indices.append(i)
print(len(remove_indices))

df=df.drop(remove_indices)
df=df.drop(['primary_site','patient_id'], axis=1)
print(df.shape)
df.columns.tolist()

# data splitting
    
# In[3]:
df.columns.tolist()
X=df.drop(['cancer_type'], axis=1)

y=df['cancer_type']


y1=y.tolist()
cancers=df.cancer_type.unique()
for i in range(len(y1)):
    for j in range(len(cancers)):
        if y1[i]==cancers[j]:
            y1[i]=j

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('The training set shape is {}, and the testing  set shape is {}'.format(x_train.shape, x_test.shape))
x_train.head()

x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# ## one-hot encoding

# In[ ]:

if categorical_var=='on':
    x_train = pd.get_dummies(x_train, columns = ['medication','sex'])
    x_test = pd.get_dummies(x_test, columns = ['medication','sex'])
    '''
    # More advanced encoders
    import category_encoders as ce
    # Polynomial Encoder
    #encoder = ce.polynomial.PolynomialEncoder(cols=[''])
    # Backward Difference Encoder
    encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=[''])
    encoder.fit(df, verbose=1)
    df = encoder.transform(df)
    df = df.drop(['intercept'], axis=1)
    print(df.shape)
    '''

columns=df.columns.tolist()


print(x_train.head())
print(x_test.head())
print(x_train.shape)
print(x_test.shape)

# neural net

# In[1]:

print('Neural Net 1 in progress...')

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics  

seed = 7
np.random.seed(seed)

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
dummy_y2 = np_utils.to_categorical(encoded_Y)

o=dummy_y.shape[1]
def baseline_model():
	model = Sequential()
	model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(o, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=128, verbose=0)
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# In[1]:

print('Neural Net 2 in progress...')

# Define model
NN_model = Sequential()
NN_model.add(Dense(64, input_dim=x_train.shape[1], activation= "relu"))
NN_model.add(Dense(128, activation= "relu"))
NN_model.add(Dense(128, activation= "relu"))
NN_model.add(Dense(dummy_y.shape[1], kernel_initializer='normal', activation='sigmoid'))
NN_model.summary() #Print model Summary
# Compile model
NN_model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
# Fit Model
NN_model.fit(x_train, dummy_y, epochs=20)


# Predictions/probs on the test dataset
NN_predicted = pd.DataFrame(NN_model.predict(x_train))
probs = pd.DataFrame(NN_model.predict_proba(x_train))
ind1 = []
ind2 = []
for i in range(0,len(NN_predicted)):
    ind1.append(np.argmax(NN_predicted.iloc[i,:], axis=0))
    ind2.append(np.argmax(probs.iloc[i,:], axis=0))
    for j in range(NN_predicted.shape[1]):
        NN_predicted.iloc[i,j] = 1 if j==ind1[i] else 0
        probs.iloc[i,j] = 1 if j==ind2[i] else 0


# Store metrics
NN_accuracy = metrics.accuracy_score(dummy_y, NN_predicted)
NN_roc_auc = metrics.roc_auc_score(dummy_y, probs)

print('Accuracy : {:3.3f}, AUC : {:3.3f}'.format(NN_accuracy, NN_roc_auc))

# for checking the classification accuracy of each class
dict0 = {}
for i in range(dummy_y.shape[0]):
        j = np.argmax(dummy_y[i,:], axis=0)
        if j not in dict0:
            dict0[j] = 1
        else:
            dict0[j] += 1

dict1 = {}
for i in range(NN_predicted.shape[0]):
    if NN_predicted.iloc[i,:].tolist() == list(dummy_y[i,:]):
        j = np.argmax(NN_predicted.iloc[i,:], axis=0)
        if j not in dict1:
            dict1[j] = 1
        else:
            dict1[j] += 1



# ## XGBoost

# In[ ]:


df.columns.tolist()
X=df.drop(['cancer_type'], axis=1)

y=df['cancer_type']

y1=y.tolist()
cancers=df.cancer_type.unique()
for i in range(len(y1)):
    for j in range(len(cancers)):
        if y1[i]==cancers[j]:
            y1[i]=j

x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=42)

print('The training set shape is {}, and the testing  set shape is {}'.format(x_train.shape, x_test.shape))
x_train.head()

x_train.reset_index(drop=True, inplace=True)
#y_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
#y_test.reset_index(drop=True, inplace=True)

if categorical_var=='on':
    x_train = pd.get_dummies(x_train, columns = ['medication','sex'])
    x_test = pd.get_dummies(x_test, columns = ['medication','sex'])
    '''
    # More advanced encoders
    import category_encoders as ce
    # Polynomial Encoder
    #encoder = ce.polynomial.PolynomialEncoder(cols=[''])
    # Backward Difference Encoder
    encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=[''])
    encoder.fit(df, verbose=1)
    df = encoder.transform(df)
    df = df.drop(['intercept'], axis=1)
    print(df.shape)
    '''

columns=df.columns.tolist()


print(x_train.head())
print(x_test.head())
print(x_train.shape)
print(x_test.shape)

print('XGBoost in progress...')

#Import libraries:
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from sklearn import cross_validation, metrics   #Additional scklearn functions

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, x_train, y_train, x_test, y_test, useTrainCV=False, cv_folds=5, early_stopping_rounds=10):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train.values, label=y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    #Predict training set:
    #x_train_predictions = alg.predict(x_train)
    #x_train_predprob = alg.predict_proba(x_train)[:,1]

    #Print model report:
    #print("\nModel Report")
    #print("Accuracy (Train) : %.4g" % metrics.accuracy_score(y_train.values, x_train_predictions))
    #print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, x_train_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    # Predictions/probs on the test dataset
    #xgboost_predicted = pd.DataFrame(alg.predict(x_test))  
    # Store metrics
    #xgboost_accuracy = metrics.accuracy_score(y_test, xgboost_predicted)  
    #print('Accuracy (Test) =',xgboost_accuracy)

xgb1 = XGBClassifier(
learning_rate=0.01,
n_estimators=300,
max_depth=20,
min_child_weight=1,
gamma=0.1,
reg_alpha=3,
subsample=0.7,
objective = "multi:softprob",
colsample_bytree=0.5,
nthread=4,
scale_pos_weight=1,
seed=27)
modelfit(xgb1, x_train, y_train, x_test, y_test)

# Predictions/probs on the test dataset
xgboost_predicted = pd.DataFrame(xgb1.predict(x_test))  
# Store metrics
xgboost_accuracy = metrics.accuracy_score(y_test, xgboost_predicted)  

# Predictions/probs on the test dataset
# >>>>>>>>>>>>>>>>>>>>>>>>> predicted = pd.DataFrame(///.predict(x_test))
probs = pd.DataFrame(xgb1.predict_proba(x_test))
#xgboost_roc_auc = metrics.roc_auc_score(y_test, probs[1])
xgboost_confus_matrix = metrics.confusion_matrix(y_test, xgboost_predicted)  
xgboost_classification_report = metrics.classification_report(y_test, xgboost_predicted)  

print('Accuracy : {:3.3f}'.format(xgboost_accuracy))

