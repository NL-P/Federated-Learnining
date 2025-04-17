import os
import torch
import numpy as np
import logging.config
import matplotlib.pyplot as plt
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
ILs=[1,2,3,4,5,6]
M1=107
NUM_CLIENTS = len(ILs)

# DATASET = "fashion_mnist"
DATASET = "caisson"

tons = ['Intact','Damage1', 'Damage2', 'Damage3']

path = "...\\Noised\\"          #Place your directory here

noise_percents=range(0,6)

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
for time in range(M1,M1+1):
  
    if __name__=="__main__":
        if not os.path.isdir('models'):
            os.mkdir('models')
        if not os.path.isdir('results'):
            os.mkdir('results')
        
        #Initialize a logger to log epoch results
        logname = ('results/log_federated_' + DATASET + "_" + str(NUM_EPOCHS) +"_"+ str(NUM_CLIENTS) + "_" + str(LOCAL_ITERS))
        logging.basicConfig(filename=logname,level=logging.DEBUG)
        logger = logging.getLogger()
    
        #global_model = CNN(input_shape, n_classes)
        LAYERS = [
            tf.keras.layers.Conv1D(filters=4, kernel_size=6, activation='relu', input_shape=(X_train.shape[2],X_train.shape[3])),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(filters=4, kernel_size=4, strides=1, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ]
        global_model = tf.keras.models.Sequential(LAYERS)      
        global_model.save_weights("models/"+ DATASET + "_" + str(NUM_CLIENTS) + "_federated.h5")
    
        LOSS = "categorical_crossentropy"
        OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        
        #optimizer.build(global_model.trainable_variables)
    
        global_model.compile(loss=LOSS , optimizer=OPTIMIZER, metrics=['accuracy'])
        
        # Train the model
        
        history = global_model.fit(X_train[0], y_train[0],batch_size=BATCH_SIZE, epochs=LOCAL_ITERS, verbose=1, validation_data=(X_val[0],y_val[0]), shuffle=True)
        
        global_model.save_weights("models/" + DATASET + "_1_" + str(NUM_CLIENTS) + "_federated.h5")
        
        all_train_loss_1, all_train_loss_2, all_train_loss_3, all_train_loss = [],[],[],[]
        all_val_loss_1,   all_val_loss_2,   all_val_loss_3,   all_val_loss   = [],[],[],[]
        val_loss_min = np.inf
        
        all_val_accuracy = []
        
        OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        
    # Initialize lists for tracking per-client and global loss
    all_train_loss_clients = [[] for _ in range(NUM_CLIENTS)]
    all_val_loss_clients = [[] for _ in range(NUM_CLIENTS)]
    all_train_loss = []
    all_val_loss = []
    
    # Train the model for the given number of epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch:", epoch)
        
        local_params = []
        local_model = tf.keras.models.clone_model(global_model)
        local_model.load_weights(f"models/{DATASET}_1_{NUM_CLIENTS}_federated.h5")
    
        # Train each client
        for idx in range(NUM_CLIENTS):
            local_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
            local_model.fit(X_train[idx], y_train[idx], epochs=LOCAL_ITERS, batch_size=BATCH_SIZE,
                            verbose=1, validation_data=(X_val[idx], y_val[idx]), shuffle=True)
    
            # Store local weights
            local_weights = local_model.get_weights()
            local_params.append(local_weights)
    
        # Perform federated averaging
        global_params = FedAvg(local_params)
        global_model.set_weights(global_params)
    
        # Evaluate per-client losses
        epoch_train_losses = []
        epoch_val_losses = []
    
        for idx in range(NUM_CLIENTS):
            train_loss, train_acc = global_model.evaluate(X_train[idx], y_train[idx], verbose=0)
            val_loss, val_acc = global_model.evaluate(X_val[idx], y_val[idx], verbose=0)
    
            all_train_loss_clients[idx].append(train_loss)
            all_val_loss_clients[idx].append(val_loss)
    
            epoch_train_losses.append(train_loss)
            epoch_val_losses.append(val_loss)
    
        # Average loss across all clients
        train_loss_avg = sum(epoch_train_losses) / NUM_CLIENTS
        val_loss_avg = sum(epoch_val_losses) / NUM_CLIENTS
    
        all_train_loss.append(train_loss_avg)
        all_val_loss.append(val_loss_avg)
    
        global_model.save_weights(f"models/{DATASET}_1_{NUM_CLIENTS}_federated.h5")
    
        # Save best model
        if val_loss_avg < val_loss_min:
            val_loss_min = val_loss_avg
            global_model.save(f'CNN_model_10/Federated_CNN_{time}_best')

    # Load the best model
    global_model.load_weights(f"models/{DATASET}_{NUM_CLIENTS}_federated.h5")
    
    # Evaluate on test set (example: only first client)
    test_loss, test_acc = global_model.evaluate(X_test[0], y_test[0])
    logger.info('Test accuracy: {:.8f}'.format(test_acc))
    
    # Prepare results for all clients
    loss_data = {'Epoch': list(range(1, NUM_EPOCHS + 1))}
    
    for idx in range(NUM_CLIENTS):
        loss_data[f'Validation Loss {idx+1}'] = all_val_loss_clients[idx]
        loss_data[f'Training Loss {idx+1}'] = all_train_loss_clients[idx]
    
    loss_data['Validation Loss'] = all_val_loss
    loss_data['Training Loss'] = all_train_loss
    
    loss_df = pd.DataFrame(loss_data)

    
    loss_df.to_excel('CNN_model_10\\Federated_CNN_{}.xlsx'.format(time))
    global_model.save('CNN_model_10\\Federated_CNN_{}'.format(time))
   
#%% Plotting validation accuracy
for i in range(NUM_CLIENTS):
    plt.plot(range(1, NUM_EPOCHS + 1), all_val_loss_clients[i], label=f'Validation Loss {i+1}')
    plt.plot(range(1, NUM_EPOCHS + 1), all_train_loss_clients[i], label=f'Training Loss {i+1}')

plt.plot(range(1, NUM_EPOCHS + 1), all_val_loss, label='Average Validation Loss', linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), all_train_loss, label='Average Training Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Validation and Training Loss during Model {time}')
plt.legend()
plt.show()

#%%
def inv_Transform_result(y_pred):    
    y_pred = y_pred.argmax(axis=1)
    y_pred = encoder.inverse_transform(y_pred)
    return y_pred

global_model = keras.models.load_model('CNN_model_10\\Federated_CNN_{}_best'.format(time))

y_pred=global_model.predict(X_test[0])
Y_pred=inv_Transform_result(y_pred)
Y_test = inv_Transform_result(y_test[0])

from sklearn.metrics import confusion_matrix

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
