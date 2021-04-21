
# GpNet: Locally connected Network for Genomic prediction
 

## Model summary
We propose a GpNet, a deep learning network for genomic prediction of beef cattle. GpNet can estimate genetic effects of each LD-block consisting of locally adjacent two or more single nucleotide polymorphisms (SNP) by locally connected layer. 

### 1. Locally connected layer
![AnyConv com__Figure2](https://user-images.githubusercontent.com/71325306/115521432-1eddaa80-a2c6-11eb-87d8-16ddeffc4e74.png)
Locally connected layer is inspired by causal convolution (Oord, et al., 2016) and local convolution (Taigman, et al., 2014). By using locally connected layer, the network cannot violate the order of SNP: the SNP at the $n$-position cannot depend on any of SNPs to the 5'-end ($$x_{n+1},x_{n+2},...,x_{E}$$).  

### 2. Network Structure
![f](https://user-images.githubusercontent.com/71325306/115522431-1043c300-a2c7-11eb-9a8c-9d1c5287adfc.png)
 GpNet consists of the stacks of locally connected layer. Both skip connection (He, et al., 2016) and relu activation (Nair and Hinton, 2010) are used throughout the network to enable training of much deeper model. GpNet can be scaled by the different layer depth $d$ and stack number $s$.
 
our models were implemented by **tensorflow 2.3** and **keras**
  
## Implementation

### 1. Preparing
The pre-trained weight must exist as `weights.h5` and in the path where `main.py` is located.
#### 1.1. Configuration
You can set the configuration  of GpNet at main.py.

```
'batch_size' : batch size,
'epochs' : train epochs
'lr' : train learning rate
'depth' : GpNet #N of locally connected layer depth
'stack' : GpNet #N of locally connected layer stacks
'gv' : genetic variance of phenotype
'rv' : residual variance of phenotype
'device' : GPU number to use
'data_load' : A or G (A : array , G : generator)
```
#### 1.2. Input Data format
Our code requires the `.raw` data format of `plink`.  See https://www.cog-genomics.org/plink2/formats#raw for more details.
**Caution:** Our code does **not** allow for **variance missing**
### 2. Execution

#### 2.1. Train GpNet
```
python main.py --raw [raw] --out [out] --mode train
```
1. [raw] : Path of input raw data.
2. [out] : Directory name for saving the trainned weights, history and prediction results. 

#### 2.2. Test GpNet
```
python main.py --raw [raw] --out [out] --mode test
```
1. [raw] : Path of input raw data.
2. [out] : Directory name for saving the prediction results. 

#### 2.3. Example
**Train sample data**
```
python main.py --raw data/sample.raw --out results --mode train
```
**Test sample data**
```
python main.py --raw data/sample.raw --out results --mode test
```

## Reference
- Oord, A.v.d._, et al._ Wavenet: A generative model for raw audio. _arXiv preprint arXiv:1609.03499_ 2016.
- Taigman, Y._, et al._ Deepface: Closing the gap to human-level performance in face verification. In, _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2014. p. 1701-1708.
- He, K._, et al._ Deep residual learning for image recognition. In, _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016. p. 770-778.
- Nair, V. and Hinton, G.E. Rectified linear units improve restricted boltzmann machines. In, _Icml_. 2010.
