import pandas as pd
import pickle
import serial
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
loaded_model_rf = pickle.load(open("finalized_model.sav", 'rb'))
loaded_model_knn = pickle.load(open("finalized_model_knn.sav", 'rb'))

ser=serial.Serial('COM5',9600)
ser.flushInput()
ser.close()
ser.open()

#this part is to make predictions from the prepared dataset
"""data_pr=pd.read_csv("datatest_07.txt",sep=" ",header=None)

col_name = ['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','E','T1','T2']
data_pr.columns = col_name

features= data_pr.drop(["E",'T1',"T2"], axis = 1)
X=features.values
output = data_pr["E"]
pred_res= loaded_model_knn.predict(X)

conf_pr= confusion_matrix(pred_res,output)
label=["0","1","2","3","4","5"]
plt.figure()
sns.heatmap(conf_pr,annot=True,xticklabels=label,yticklabels=label,fmt="g")
print(accuracy_score(pred_res,output))"""

#listem = np.zeros((21, 11),dtype="int32")
while (True):
    inline = str(ser.readline().strip())
    inline=inline.replace("'","")
    inline=inline.replace("b","")
    info=inline.split(";")[:-1]
    info = np.array(info,dtype="int32").reshape((1,-1))
    #listem = np.append(listem, info, axis=0)
    #if (len(listem)%21 ==0):
        #listem_arr = np.array(listem[-21:,:])
        #listem_arr = np.mean(listem_arr, axis=0).reshape((1,-1))
        #pred=loaded_model.predict(listem_arr)"""
    pred=loaded_model_rf.predict(info)
    if pred== 0:
        print("el açık")
    elif pred==1:
        print("yumruk")
    elif pred==2:
        print("sayı bir")   
    elif pred==3:
        print("sayı 2")
    elif pred==4:
        print("rock ")
    else :
        print("spider-man")
            
  #  if len(listem) >= 252:
   #     listem = listem[-21:,:]
        