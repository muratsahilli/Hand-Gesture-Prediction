# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:49:25 2022

@author: murat
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import librosa ,librosa.display
from scipy.fftpack import fft


def accuracies_mic(y_test,y_pred):
    # precision, recall, f1, accuracy skorları hesaplıyoruz.
    PS_mi = precision_score(y_test,y_pred,average='micro')
    RS_mi = recall_score(y_test,y_pred,average='micro')
    F1_mi = f1_score(y_test,y_pred,average='micro')
    return PS_mi,RS_mi,F1_mi
    
def accuracies_mac(y_test,y_pred):    
    PS_ma = precision_score(y_test,y_pred,average='macro')
    RS_ma = recall_score(y_test,y_pred,average='macro')
    F1_ma = f1_score(y_test,y_pred,average='macro')
    return PS_ma,RS_ma,F1_ma

def accuracies_we(y_test,y_pred):  
    PS_we = precision_score(y_test,y_pred,average='weighted')
    RS_we = recall_score(y_test,y_pred,average='weighted')
    F1_we = f1_score(y_test,y_pred,average='weighted')
    return PS_we,RS_we,F1_we

micros = []
macros = []
weights = []
# bir modeli değerlendirme
def evaluate_model(trainX, trainy, testX, testy, model):
	# modeli eğitme
    model.fit(trainX, trainy)
	# tahmin etme
    yhat = model.predict(testX)
	# tahminin doğruluğunu hesaplama
    accuracy = accuracy_score(testy, yhat)
    micros.append(accuracies_mic(testy, yhat))
    macros.append(accuracies_mac(testy, yhat))
    weights.append(accuracies_we(testy, yhat))
    return accuracy * 100.0

# modelleri değerlendirme {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models:
		# modeli değerlendirme
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# işlemi göster
		print('>%s: %.3f' % (name, results[name]))
	return results

def plot_results(mic,mac,we,name):
    
    N = 3
    ind = np.arange(N) 
    width = 0.25
      
    bar1 = plt.bar(ind, mic, width, color = 'gold')
    bar2 = plt.bar(ind+width, mac, width, color='teal')
    bar3 = plt.bar(ind+width*2, we, width, color = 'indigo')
      
    plt.xlabel(name)
    plt.title(name + "  Scores")
    
    plt.xticks(ind+width,['Precision', 'Recall', 'F1'])  
    plt.legend( (bar1, bar2, bar3), ('micro', 'macro', 'weight') )
    plt.show()


data=pd.read_csv("D:/OKUL/Biyomedikal/data06.txt",sep=" ",header=None)

col_name = ['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','E','T1','T2']
data.columns = col_name

features= data.drop(["E",'T1',"T2"], axis = 1)
output = data["E"]

print("The correlation coefficient of the variables in the data set with each other")
corr = np.abs(data.corr(method='pearson'))
plt.figure(figsize = (8,6))
sns.heatmap(corr, annot = True)

data.hist(bins=14,figsize=(16,9),grid=False);
X= data.iloc[:,:-3].values
y=data.iloc[:,-3].values

X_train,X_test,Y_train,Y_test=train_test_split(X,y, random_state=22, test_size=0.15)


arr = np.full([1, len(X)+1], 1)
for i in range(len(X[1])):
    a = X[:,i]
    lib = np.abs(librosa.stft(a.astype(float),n_fft=2, hop_length=1))
    arr= np.concatenate((arr,lib))
arr = np.delete(arr, (0), axis=0)  
plt.figure(figsize=(15, 5)) 
librosa.display.specshow(arr, hop_length=16, x_axis='time', y_axis='linear') 
plt.colorbar(format='%+2.0f dB')
plt.show()

    
#create time domain plot
sampling_rate = 21
N= len(X)
zaman = np.linspace(0, N/sampling_rate, N)

plt.plot (zaman, a)
plt.title ('Time Domain Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()

frequency = np.linspace(0.0, 21/2, int(N/2))
freq_data = fft(a)
freq_abs = 2/N * np.abs(freq_data[0:int(N/2)])

plt.plot(frequency[700:900], freq_abs[700:900])
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
# modelleri hesaplama
models_results = evaluate_models(X_train, Y_train, X_test, Y_test, models)

# 10-fold cross validation
results = []
names=[]

for name, model in models:    
    kfold = KFold(n_splits=10, shuffle=True, random_state=22)
    cv_results = np.abs(cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy'))
    results.append(cv_results)
    names.append(name)
    
puan=[]
for i in range(len(names)):
    print(names[i],results[i].mean())
    puan.append(results[i].mean())
print("En yüksek acc")
print(names[puan.index(max(puan))],max(puan))

# Models accuracy graph
mod_point = []
for i in names:
    mod_point.append(models_results[i]) 

plt.plot(names,mod_point)
plt.xlabel('models')
plt.ylabel('accuracy')
plt.title('ACCURACY GRAPH OF THE MODELS')
plt.show()

# 10-fold cross validation graph
plt.plot(names,puan)
plt.xlabel('models')
plt.ylabel('accuracy')
plt.title('ACCURACY GRAPH OF THE MODELS 10-fold cross validation')
plt.show()

#comparing precision, recall and F1 scores
for i in range(len(names)):
    plot_results(micros[i],macros[i],weights[i],names[i])

#comparing only F1 micro scores for each model
f1_micros=[]
for i in range(len(names)):
    f1_micros.append(micros[i][2])
    
plt.plot(names,f1_micros)
plt.xlabel('models')
plt.ylabel('F1')
plt.title("MODEL's f1 micro SCORES")
plt.show()


rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
y_pred = rf.predict(X_test)

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))

conf= confusion_matrix(y_pred,Y_test)
label=["0","1","2","3","4","5"]
plt.figure()
sns.heatmap(conf,annot=True,xticklabels=label,yticklabels=label,fmt="g")

knn_model = KNeighborsClassifier()
knn_model.fit(X_train,Y_train)
y_pred_knn = knn_model.predict(X_test)

conf_knn= confusion_matrix(y_pred_knn,Y_test)
label=["0","1","2","3","4","5"]
plt.figure()
sns.heatmap(conf_knn,annot=True,xticklabels=label,yticklabels=label,fmt="g")

filename = 'finalized_model_knn.sav'
pickle.dump(knn_model, open(filename, 'wb'))

namesdf=pd.DataFrame(names)
microsdf=pd.DataFrame(micros)
macrosdf=pd.DataFrame(macros)
weightsdf=pd.DataFrame(weights)
concat_df  = pd.concat([namesdf,microsdf,macrosdf,weightsdf], axis=1)
concat_df.columns=["model_names","presicion_micro","presicion_macro","presicion_weighted",
                   "recall_micro","recall_macro","recall,weighted",
                   "F1_micro","F1_macro","F1,weighted"]

# plot the dataframe
concat_df.plot(x="model_names", y=["presicion_micro","presicion_macro","presicion_weighted",
                   "recall_micro","recall_macro","recall,weighted",
                   "F1_micro","F1_macro","F1,weighted"], kind="bar", figsize=(30,10))
plt.legend(loc="lower center")
# print bar graph
plt.show()

concat_df.to_csv('comparison.csv')
