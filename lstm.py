import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
import numpy as np

def create_label(label_all,label):
    current_label=[]
    for l in label_all:
        if l==label:
            current_label.append(1)
        else:
            current_label.append(0)
    return current_label

def read_data(data_dir,data_div):
    all_data=[]
    all_label_orig=[]
    for line in open(data_dir):
        if line.find("?")!=-1:
            continue
        line=line[:-1] #removing the new line symbol
        row=line.split(',')
        row=[float(i) for i in row]
        all_label_orig.append(row[len(row)-1])
        #row.pop(0)#to remove first element
        all_data.append(row[:-1])

    label_u=[]
    for label in all_label_orig:
        #print(data)
        found=False
        for l in label_u:
            if l==label:
                found=True
                break
        if found!=True:
            label_u.append(label)
    
    all_label=[]
    for label in all_label_orig:
        all_label.append(create_label(label_u,label))

    train_data,test_data,train_label,test_label=train_test_split(all_data,all_label,test_size=data_div,random_state=42)
    
    #print("train_data:",len(train_data))
    #print("train_label: ",len(train_label))
    #print("test_data: ",len(test_data))
    #print("test_label: ",len(test_label))
    
    return np.array(train_data),np.array(train_label),np.array(test_data),(test_label),len(label_u)

def create_network(data_arr,reshaped_training_data,no_of_classes):
    model = Sequential()
    model.add(LSTM(45,input_shape=(reshaped_training_data.shape[1],1),activation='relu',return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(45,activation='relu',return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(int(len(data_arr[0])/2),activation='relu',return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(no_of_classes,activation='softmax'))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model

def train_and_test(dataset_dir,display_each):
    train_data,train_label,test_data,test_label,no_of_classes=read_data("../processed_data_2/"+dataset_dir,0.5)
    reshaped_training_data = np.reshape(train_data, (train_data.shape[0],train_data.shape[1],1))
    reshaped_test_data = np.reshape(test_data, (test_data.shape[0],test_data.shape[1],1))
    model=create_network(train_data,reshaped_training_data,no_of_classes)

    model.fit(reshaped_training_data,train_label,epochs=500,batch_size=int(len(train_data)),verbose=2)#2

    predicted_result=model.predict(reshaped_test_data)
    rounded_predicted_result=[]
    correct=0
    a=0
    while a<len(predicted_result):
        b=0
        all_ok=True
        round_result=[]
        while b<len(predicted_result[a]):
            if test_label[a][b]!=round(predicted_result[a][b]):
                round_result.append(round(predicted_result[a][b]))
                all_ok=False
            b+=1
        rounded_predicted_result.append(round_result)
        a+=1
        if all_ok==True:
            correct+=1

    print(dataset_dir+' Total Accuracy: %.2f' % (correct/len(test_data)*100))

file_list=os.listdir("../processed_data_2/")
for file in file_list:
    print(file)
    train_and_test(file,True)