## Deep Ranking Distribution Preserving Hashing For Robust Multi-Label Cross-modal Retrieval -- Tensorflow Implementation

### Deep Ranking Distribution Preserving Hashing (RDPH) 

The details can be found in the paper Deep Ranking Distribution Preserving Hashing For Robust Multi-Label Cross-modal Retrieval  (submitted to IEEE Transactions on Multimedia)

#### Implementation platform: 
* python 3.7  
* tensorflow 2.5.0 
* matlab 2016b

#### Datasets
We use the following datasets.

* MIRFlickr-25k  
* MSCOCO
* NUSWIDE

Pre-extracted features by CLIP:
* MIRFlickr-25k [download](https://pan.baidu.com/s/1zh1S8hr6Ac_Ky_xRYF-jzw) ah6j 


#### Training
The command for training is
* python3 train_RDPH.py
* our trained model (on MIRFlickr-25k) can be [download](https://pan.baidu.com/s/1tu2LE08eDBqZMD_sOEyIEQ) y2xw
#### Evaluation
The command for evaluation is
* extract hash codes: python3 RDPH_encoding.py
* run evaluation.m