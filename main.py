##2020. 11. 09.
##2021. 04. 21.

import argparse
import os
import numpy as np
from GP_Net import *
from utils import *
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger

parser = argparse.ArgumentParser()
parser.add_argument("--raw", help="Path of input raw data")
parser.add_argument("--out", help="directory name for saving the prediction results")
parser.add_argument("--mode", help="train or test", default = 'test')
args = parser.parse_args()

assert os.path.isfile(args.raw), args.raw + " dosen't exist"
assert os.path.isdir(args.out), args.out + " dosen't exist"
assert args.mode in ['train','test'], "--mode must be included in ['train','test']"

#####################Config#######################

config = {
'batch_size' : 2, 
'epochs' : 10, # train epochs
'lr' : 1e-4, # train learning rate
'depth' : 4, # GpNet #N of locally connected layer depth
'stack' : 1, # GpNet #N of locally connected layer stacks
'gv' : 1, # genetic variance of phenotype
'rv' : 1, # residual variance of phenotype
'device' : '0', # GPU number to use 
'data_load' : 'G' # A : array , G : generator
}
####################################################

print('#'*15+'CONFIGURATION'+'#'*15+'\n')
for k, i in config.items(): print('{0}: {1}'.format(k,i))
print('\n'+'#'*43+'\n')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=config['device']

#Load data
print('Load Data set ......')
if args.mode == 'test':
    data = data_load(args.raw, config['batch_size'], mode = config['data_load'], shuffle = False)
else:
    data = data_load(args.raw, config['batch_size'], mode = config['data_load'], shuffle = True)

#Load model
model = GP_Net(marker_den = data.den, 
            activation = 'relu', 
            depth = config['depth'], 
            stacks = config['stack'])

if os.path.isfile('weights.h5'):
    model.load_weights('weights.h5')

#Implement
if args.mode == 'train':

    logger = CSVLogger(args.out+"/history.txt", append=True, separator='\t')

    loss = sh_mse(config['gv'],config['rv'])
    adam = optimizers.Adam(lr=config['lr'])

    model.compile(loss=loss, metrics = [Pearson_cor], optimizer = adam)

    #Training
    print('######################')
    print('##                  ##')
    print('##   Train GP-Net   ##')
    print('##                  ##')
    print('######################')
    
    if config['data_load'] == 'G':
        
        model.fit(data.G, 
            batch_size = config['batch_size'], epochs = config['epochs'], 
            callbacks= logger, steps_per_epoch = len(data.G))
        
        
    else:
        model.fit(data.X,data.y, 
            batch_size = config['batch_size'], epochs = config['epochs'], 
            callbacks= logger)

    model.save_weights(args.out+'/weights.h5')
    
if config['data_load'] == 'G':
        
    pred = np.array([0])
    true = np.array([0])
    
    print('Prediction NOW begin...')
    
    for _ in tqdm.tqdm(range(len(data.G))):
        
        batch_X, batch_y = next(data.G)
         
        gebv = model.predict(batch_X)
        true = np.concatenate([true,batch_y])
        pred = np.concatenate([pred,gebv[:,0]])
        
    true = true[1:]
    pred = pred[1:]

else:
        
    true = data.y
    pred = model.predict(data.X, verbose = 1)[:,0]

save_results(data, true, pred, config, args.out)

