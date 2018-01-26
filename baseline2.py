import pandas as pd
import numpy.random as nr
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn import ensemble

from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder


#funkcija koja traži najbolje hiperparametre gradient boostinga
def gbHyperparametars(X_train, y_train):
    algorithmParameters = dict() #dict hiperparametara koje ćemo testirati

    #popunjavanje algorithmParameters
    nEstimatorsValues = []
    for i in range(6, 12):
        nEstimatorsValues.append(2**i)

                    
    algorithmParameters['n_estimators'] = nEstimatorsValues

        
    learningRateValues  = []
    for i in range(3):
        learningRateValues.append(0.01 * 10**i)

    algorithmParameters['learning_rate'] = learningRateValues 


    algorithmParameters['max_depth'] = [2, 3, 4]

        
    model = ensemble.GradientBoostingClassifier()

    #traženje i ispis najboljih hiperparametara
    grid = GridSearchCV(model, algorithmParameters)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.n_estimators)
    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.learning_rate)


        
    classifierGB = ensemble.GradientBoostingClassifier(n_estimators = grid.best_estimator_.n_estimators, max_depth = grid.best_estimator_.max_depth, 
                                                learning_rate = grid.best_estimator_.learning_rate)

    return classifierGB

#funkcija koja traži najbolje hiperparametre random forest
def rfHyperparametars(X_train, y_train, n_estimators):

    algorithmParameters = dict()

    
    algorithmParameters['min_samples_split'] = [2, 3, 4, 8]

    algorithmParameters['max_features'] = [0.01, 0.05, 0.1]

    #algorithmParameters['max_depth'] = [16, 32, None] 

        
    model = ensemble.RandomForestClassifier(n_estimators = n_estimators)

    grid = GridSearchCV(model, algorithmParameters)
    grid.fit(X_train, y_train)

    #print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.min_samples_split)
    print(grid.best_estimator_.max_features)


    classifierRF = ensemble.RandomForestClassifier(n_estimators = n_estimators, min_samples_split = grid.best_estimator_.min_samples_split,
                                                    max_depth = None, max_features = grid.best_estimator_.max_features)


    return classifierRF

#hiperparametri za logističku regresiju
def lrHyperparametars(X_train, y_train):

    algorithmParameters = dict()
                
    algorithmParameters['penalty'] = ['l1', 'l2']

        
    CValues   = []
    for i in range(1, 15):
        CValues.append(0.001 * 2**i)

    algorithmParameters['C'] = CValues  


    #algorithmParameters['dual'] = [True, False]

        
    model = linear_model.LogisticRegression()

    grid = GridSearchCV(model, algorithmParameters)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.penalty)
    #print(grid.best_estimator_.dual)
    print(grid.best_estimator_.C)



    classifierLR = linear_model.LogisticRegression(penalty = grid.best_estimator_.penalty, C = grid.best_estimator_.C)
    #dual = grid.best_estimator_.dual)

    return classifierLR



def getUserDescription(df_train, df_test, featureName = 'description'):

    userTrain = df_train.user

    userTextTrain = []

    for dictionary in userTrain:
        
        temp = ''
        
        for key, value in dictionary.items():
                
            if key == featureName:
                temp = value
                            
        userTextTrain.append(temp)
                            
    userTextTrain = np.asarray(userTextTrain)    



    userTest = df_test.user

    userTextTest = []

    for dictionary in userTest:
        
        temp = ''
        
        for key, value in dictionary.items():
                
            if key == featureName:
                temp = value
                            
        userTextTest.append(temp)
                            
    userTextTest = np.asarray(userTextTest)  

    return userTextTrain, userTextTest

def getHashtag(df_train, df_test):
    
    firstEntities = df_train.entities

    hashtagsArray = []

    for dictionary in firstEntities:
        temp = ''

        for key, values in dictionary.items():

            if key == 'hashtags':
            
                for i in range(values.__len__()):
                    
                    if isinstance(values[i], dict):
                        for keyTemp, valueTemp in values[i].items():
                            if keyTemp == 'text':
                                
                                temp = valueTemp

        hashtagsArray.append(temp)

    
    hashtagsArray = np.asarray(hashtagsArray)


    firstEntities = df_test.entities

    hashtagsArrayTest = []

    for dictionary in firstEntities:
        temp = ''

        for key, values in dictionary.items():

            if key == 'hashtags':
            
                for i in range(values.__len__()):
                    
                    if isinstance(values[i], dict):
                        for keyTemp, valueTemp in values[i].items():
                            if keyTemp == 'text':
                                
                                temp = valueTemp

        hashtagsArrayTest.append(temp)

    
    hashtagsArrayTest = np.asarray(hashtagsArrayTest)


    return hashtagsArray, hashtagsArrayTest
        


def create_output_file(filename, predictions, sentences, labels = []):

    if labels.__len__() == 0:
        df = pd.DataFrame({'Tekst' : sentences, 'Predikcija' : predictions})
    else:
        df = pd.DataFrame({'Tekst' : sentences, 'Predikcija' : predictions, 'Ispravne kategorije' : labels})

    df.to_csv(filename + '.csv')


    

df1 = pd.read_json('C:\\Users\\viktor\\Downloads\\hs_data\\share\\sexism.json', lines=True)
df2 = pd.read_json('C:\\Users\\viktor\\Downloads\\hs_data\\share\\racism.json', lines=True)
df3 = pd.read_json('C:\\Users\\viktor\\Downloads\\hs_data\\share\\neither.json', lines=True)

frames = [df1, df2, df3]

print(df1.__len__())
print(df2.__len__())
print(df3.__len__())

df = pd.concat(frames, ignore_index = True)

print(df.__len__())

from sklearn.utils import shuffle
df = shuffle(df)

del df['contributors']
del df['coordinates']
del df['extended_entities']
del df['geo']
del df['quoted_status']
del df['is_quote_status']
del df['quoted_status_id']
del df['quoted_status_id_str']
del df['retweeted']
del df['favorited']
del df['truncated']
del df['withheld_in_countries']
del df['possibly_sensitive']
del df['possibly_sensitive_appealable']
del df['id_str']
del df['in_reply_to_status_id']
del df['in_reply_to_screen_name']
del df['in_reply_to_status_id_str']
del df['in_reply_to_user_id']
del df['in_reply_to_user_id_str']

#df['Annotation'] = df['Annotation'].astype('category').cat.codes
df['lang'] = df['lang'].astype('category').cat.codes

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

sw = stopwords.words("english")
vectorizer = TfidfVectorizer(lowercase=True, stop_words = sw, binary = True, sublinear_tf  = True, norm = None)

df_train, df_test = train_test_split(df, test_size = 0.2)

#tweetovi i label
X_train = df_train['text'].as_matrix()
y_train = df_train['Annotation'].as_matrix()

X_test = df_test['text'].as_matrix()
y_test = df_test['Annotation'].as_matrix()

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()


#dodaj user description
userTextTrain, userTextTest = getUserDescription(df_train, df_test)

userTextTrain = vectorizer.fit_transform(userTextTrain).toarray()
userTextTest = vectorizer.transform(userTextTest).toarray()

X_train = np.concatenate((X_train, userTextTrain), axis=1)
X_test = np.concatenate((X_test, userTextTest), axis=1)


#dodaj favorite_count
favorite_countTrain = df_train['favorite_count'].as_matrix()
favorite_countTest = df_test['favorite_count'].as_matrix()

ss = StandardScaler()

favorite_countTrain = ss.fit_transform(favorite_countTrain)
favorite_countTest = ss.transform(favorite_countTest)

favorite_countTrain = np.reshape(favorite_countTrain, (-1, 1))
favorite_countTest = np.reshape(favorite_countTest, (-1, 1))

X_train = np.concatenate((X_train, favorite_countTrain), axis=1)
X_test = np.concatenate((X_test, favorite_countTest), axis=1)


#dodaj retweet_count
retweet_countTrain = df_train['retweet_count'].as_matrix()
retweet_countTest = df_test['retweet_count'].as_matrix()
   
retweet_countTrain = ss.fit_transform(retweet_countTrain)
retweet_countTest = ss.transform(retweet_countTest)

retweet_countTrain = np.reshape(retweet_countTrain, (-1, 1))
retweet_countTest = np.reshape(retweet_countTest, (-1, 1))

X_train = np.concatenate((X_train, retweet_countTrain), axis=1)
X_test = np.concatenate((X_test, retweet_countTest), axis=1)


mlb = LabelEncoder()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)


#dodaj user description
userTextTrain, userTextTest = getUserDescription(df_train, df_test, 'name')

userTextTrain = vectorizer.fit_transform(userTextTrain).toarray()
userTextTest = vectorizer.transform(userTextTest).toarray()

X_train = np.concatenate((X_train, userTextTrain), axis=1)
X_test = np.concatenate((X_test, userTextTest), axis=1)


#dodaj hashtags

hashTrain, hashTest = getHashtag(df_train, df_test)

hashTrain = vectorizer.fit_transform(hashTrain).toarray()
hashTest = vectorizer.transform(hashTest).toarray()

X_train = np.concatenate((X_train, hashTrain), axis=1)
X_test = np.concatenate((X_test, hashTest), axis=1)


#isprobaj svm
from sklearn.svm import LinearSVC

svc = LinearSVC(C = 0.0016)

svc.fit(X_train, y_train)

pred = svc.predict(X_test)


print(pred)
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test, average = 'macro'))

'''
#isprobaj lr i odredi joj hiperparametre
#lgr = lrHyperparametars(X_train, y_train)
lgr = linear_model.LogisticRegression(C = 0.128, penalty = 'l1')
lgr.fit(X_train, y_train)
pred = lgr.predict(X_test)
print(pred)
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test, average = 'macro'))
'''


#isprobaj rf i odredi mu hiperparametre
#rf = rfHyperparametars(X_train, y_train, 15) #None 0.1
rf = ensemble.RandomForestClassifier(n_estimators = 25, criterion="entropy", min_samples_split = 4, max_features = 0.05, n_jobs = -1)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print(pred)
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test, average = 'macro'))




'''

#isprobaj gradient boosting
gb = ensemble.GradientBoostingClassifier()

gb.fit(X_train, y_train)
pred = gb.predict(X_test)
print(pred)
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test, average = 'micro'))
'''

pred = mlb.inverse_transform(pred)

create_output_file("testAll", pred, df_test['text'].as_matrix(), mlb.inverse_transform(y_test))




