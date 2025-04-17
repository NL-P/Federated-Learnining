import os
import torch
import numpy as np
import logging.config
#from tqdm import tqdm
#from model import CNN
import matplotlib.pyplot as plt
#from dataset import load_dataset, visualize_dataset
import copy
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from torchvision import datasets, transforms
import seaborn as sns
import keras
#%% Input value
BATCH_SIZE=1
batch_size=1
NUM_EPOCHS = 200
LOCAL_ITERS = 1
ILs=[1,2,3,4,5,6] #,7,8,9,10,11,12
M1=107
NUM_CLIENTS = len(ILs)

# DATASET = "fashion_mnist"
DATASET = "caisson"

tons = ['Intact','Damage1', 'Damage2', 'Damage3']

path = "...\\Noised\\"          #Place your directory here

noise_percents=range(20,21)

assembles_all = list(range(1,6))
assembles_train = assembles_all[2:]
assembles_val = assembles_all[1:2]
assembles_test = assembles_all[:1]
   
# Define ton label mapping
ton_label_mapping = {
    'Intact': 1,
    'Damage1': 2, 'Damage2': 3, 'Damage3': 4,

}

#%% Input data -- Training data
X,Y,X_train,y_train=[],[],[],[]

for IL in ILs:
    for ton in tons:
        for noise_percent in noise_percents:
            for assemble in assembles_train:
                noise_assembles=10 # Input the emsemble using for training here
                if noise_percent==0 :
                    noise_assembles=1
                for noise_assemble in range(noise_assembles):  # Repeat 10 times for each noise level
                    df = pd.read_csv(os.path.join(path, 'SVD_15ACCs_Caisson{}_{}_noise_{}_asb_{}.csv'.format(IL,ton, noise_percent,noise_assemble+1)))
                    y = ton_label_mapping[ton]
                    Y.append(y)
                    X.append(df)
              
    n_classes = 4    #Number of classes                   
       
    X=np.array(X)
    X=X.reshape((X.shape[0],-1,1))
    
    Y=np.array(Y)
    encoder= LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    OHE_Y = to_categorical(encoded_Y)
    X_train.append(X)
    y_train.append(OHE_Y)
    X, Y = [], []
X_train = np.array(X_train)
y_train = np.array(y_train)
#%% Input data -- Validation data
X,Y,X_val,y_val=[],[],[],[]
for IL in ILs:
    for ton in tons:
        for noise_percent in noise_percents:
            for assemble in assembles_val:
                noise_assembles=10 # Input the emsemble using for training here
                if noise_percent==0 :
                    noise_assembles=1
                for noise_assemble in range(noise_assembles):  # Repeat 10 times for each noise level
                    df = pd.read_csv(os.path.join(path, 'SVD_15ACCs_Caisson{}_{}_noise_{}_asb_{}.csv'.format(IL,ton, noise_percent,noise_assemble+1)))
                    y = ton_label_mapping[ton]
                    Y.append(y)
                    X.append(df)
            
       
    n_classes = 4    #Number of classes                   
       
    X=np.array(X)
    X=X.reshape((X.shape[0],-1,1))
    
    Y=np.array(Y)
    encoder= LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    OHE_Y = to_categorical(encoded_Y)
    X_val.append(X)
    y_val.append(OHE_Y)
    X, Y = [], []
X_val = np.array(X_val)
y_val = np.array(y_val)
#%% Input data -- Testing data
X,Y,X_test,y_test=[],[],[],[]
for IL in ILs:
    for ton in tons:
        for noise_percent in noise_percents:
            for assemble in assembles_test:
                noise_assembles=10 # Input the emsemble using for training here
                if noise_percent==0 :
                    noise_assembles=1
                for noise_assemble in range(noise_assembles):  # Repeat 10 times for each noise level
                    df = pd.read_csv(os.path.join(path, 'SVD_15ACCs_Caisson{}_{}_noise_{}_asb_{}.csv'.format(IL,ton, noise_percent,noise_assemble+1)))
                    y = ton_label_mapping[ton]
                    Y.append(y)
                    X.append(df)
            
       
    n_classes = 4    #Number of classes                   
       
    X=np.array(X)
    X=X.reshape((X.shape[0],-1,1))
    
    Y=np.array(Y)
    encoder= LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    OHE_Y = to_categorical(encoded_Y)
    X_test.append(X)
    y_test.append(OHE_Y)
    X, Y = [], []
X_test = np.array(X_test)
y_test = np.array(y_test)

#%% Transforming the Input data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

#Define dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#%% Defined function
def FedAvg(params):
    """
    Average the paramters from each client to update the global model
    :param params: list of paramters from each client's model
    :return global_params: average of paramters from each client
    """
    global_params_1 = copy.deepcopy(params)
    global_params=copy.deepcopy(params[0])
    for param in range(len(params[0])):
        # Initialize sum of parameters for averaging
        param_sum = np.zeros_like(global_params_1[0][param])
        
        # Sum up the parameters from each client
        for key in range(len(params)):
            param_sum += global_params_1[key][param]
        
        # Compute average by dividing the sum by the total number of clients
        global_params[param] = param_sum / len(params)
    return global_params

#%%
from sklearn.metrics import confusion_matrix, accuracy_score
for time in range(M1,M1+1):  
    accs = []  # initialize list to store accuracy for each client

    #%%
    def inv_Transform_result(y_pred):    
        y_pred = y_pred.argmax(axis=1)
        y_pred = encoder.inverse_transform(y_pred)
        return y_pred
    
    global_model = keras.models.load_model('CNN_model_10\\Federated_CNN_{}_best'.format(time))
    for idx in range(NUM_CLIENTS):
        y_pred=global_model.predict(X_test[idx])
        Y_pred=inv_Transform_result(y_pred)
        Y_test = inv_Transform_result(y_test[idx])
        
        #  Compute and store accuracy
        acc = accuracy_score(Y_test, Y_pred)
        accs.append(acc * 100)  # optional: store as percentage
        
        plt.figure(figsize=(4,4))
        cm = confusion_matrix(Y_test, Y_pred,normalize=None)
        f = sns.heatmap(cm, annot=True,fmt='d',xticklabels=encoder.classes_,yticklabels=encoder.classes_, cmap="Blues",cbar=False)
        # Set axis labels
        plt.xlabel("Predicted Value")
        plt.ylabel("Actual Value")
        
        # Set tick labels
        
        tick_labels = ["U", "D1.1", "D1.2", "D1.3"]
           
        f.set_xticklabels(tick_labels)
        f.set_yticklabels(tick_labels)
        plt.show()
    
print("acc1=", accs ,";")   
print("Mean_acc1= {:.2f}".format(np.mean(accs)),";") 
