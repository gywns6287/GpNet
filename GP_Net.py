from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
    

class LCNN(Layer):

    def __init__(self, filters = 1, kernel_size = 2, activation = None, padding = 'L',**kwargs):
        super().__init__(**kwargs)
        self.layer = LocallyConnected1D(
                                    filters = filters, 
                                    kernel_initializer = 'he_normal',
                                    strides = 1,
                                    kernel_size = kernel_size, 
                                    activation = activation,
                                    implementation = 3)
        
        if padding == 'L':
            self.padding = ZeroPadding1D((kernel_size - 1,0))
        elif padding == 'R':
            self.padding = ZeroPadding1D((0,kernel_size - 1))
    
    def call(self, x):
        x = self.padding(x)
        x = self.layer(x)
        return x 

def GP_Net_block(x, depth = 1, padding = 'L', activation = None):

    for d in range(depth):
        
        skip = x
                
        x = LCNN(filters = 1, 
                kernel_size = 2+d,
                padding = padding,
                activation = activation)(x)
        
        x = Add()([skip,x])
    
    return x

def GP_Net(marker_den = None, activation = None,  depth = 1, stacks = 1):
    
    inputs = Input(shape = (marker_den, 1))
    x = inputs

    for s in range(stacks):
      
        x = GP_Net_block(x, depth = depth, padding = 'L', activation = activation)
    
    x = Flatten()(x)
    out = Dense(1)(x)
    
    return Model(inputs,out)

