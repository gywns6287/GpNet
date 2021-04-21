import tqdm
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os

class data_load():

    def __init__(self, data, batch_size = 16, mode = 'A', shuffle = True):
        
        assert mode in ['G','A'], "mode must be included in ['G' or 'A']"
        
        self.data_path = data
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle

        if mode == 'A':
            self.load_data_to_array()

        elif mode == 'G':
            self.load_data_to_generater()

    def load_data_to_array(self):
        
        raw  = np.loadtxt(self.data_path,dtype = str)
        self.animals = raw[1:,1]
        
        self.snps = raw[0,6:]
        self.den = len(self.snps)

        print('A Total of {0} animals in data set'.format(len(self.animals)))
        print('A Total of {0} markers in data set'.format(self.den))

        self.X = (raw[1:,6:].astype(np.float32) + 1)/3.
        self.y = raw[1:,5].astype(np.float32)

    def load_data_to_generater(self):
        
        os.system('mkdir _generator')
        print('Split dataset for generator')
        
        with open(self.data_path) as raw:
            
            self.snps = next(raw).split()[6:]
            self.den = len(self.snps)
            animals = []

            for line in raw:

                animal = line.split()[1]
                
                split_file = open(os.path.join('_generator',animal),'w') 
                split_file.write(line)
                split_file.close()

                animals.append(animal)

        self.animals = np.array(animals)

        print('A Total of {0} animals in data set'.format(len(self.animals)))
        print('A Total of {0} markers in data set'.format(self.den))

        self.G = call_generator(self.animals, self.batch_size, self.shuffle)
        
class call_generator:

    def __init__(self,animals,batch_size,shuffle):

        self.animals = animals
        self.batch = batch_size
        self.shuffle = shuffle

        self.cur_batch = 0
        self.animal_idx = list(range(len(self.animals))) 
        if self.shuffle:
            random.shuffle(self.animal_idx)

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.cur_batch == len(self):

            self.cur_batch = 0
            if self.shuffle:
                random.shuffle(self.animal_idx)
          
        batch_X = []
        batch_y = []
        
        s_idx = self.cur_batch*self.batch
        e_idx = s_idx + self.batch

        for idx in self.animal_idx[s_idx:e_idx]:

            animal = self.animals[idx]

            with open(os.path.join('_generator',animal)) as raw:
                line_ = next(raw).split()
                
            batch_X.append(line_[6:]) 
            batch_y.append(line_[5])

        else:

            self.cur_batch += 1

        batch_X = (np.array(batch_X).astype(float)+1)/3.
        batch_y = np.array(batch_y).astype(float)
        
        return  batch_X, batch_y
    
    def __len__(self):
        return math.ceil(len(self.animals)/self.batch) 



def Pearson_cor(true,pred):
        import tensorflow.keras.backend as K
        mean_true = K.mean(true)
        mean_pred = K.mean(pred)
       
        de_true = true - mean_true
        de_pred = pred - mean_pred
    
        ss_true = K.sum(K.square(de_true)) 
        ss_pred = K.sum(K.square(de_pred)) 
       
        cov = K.sum(de_true * de_pred) + 1e-16
        return (cov)/((K.sqrt(ss_true * ss_pred))+1e-16)


def sh_mse(gv,rv):
    
    h = gv/(gv+rv)

    def func(true,pred):
        mse = tf.keras.losses.MeanSquaredError()(true*h,pred)
        return mse

    return func


def save_results(data, true, pred, config, out):

    with open(out+'/validation.sol','w') as save:

        save.write('Animals\tTrue\tPred\n')

        for a, t, p in zip(data.animals,true,pred):

            save.write('\t'.join([a,str(t),str(p)])+'\n')
  
    with open(out+'/summary.txt','w') as save:

        for i, v in config.items():
            save.write('{0}: {1}\n'.format(i,v))

        save.write('-------------------\n')
        cor = np.corrcoef(true,pred)[0,1]
        h = math.sqrt(config['gv']/(config['rv']+config['gv']))
        acc = cor/h
        save.write('Cor: {0}\n'.format(cor))
        save.write('Cor/H: {0}\n'.format(acc))
        save.write('-------------------\n')

        save.write('CONFIG\n')
        for k, i in config.items(): save.write('{0}: {1}\n'.format(k,i))



