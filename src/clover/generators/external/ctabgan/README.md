# ctabgan-plus

## Official repository
Three files were downloaded from github:
https://github.com/Team-TUD/CTAB-GAN-Plus

These files are:
ctabgan_synthesizer.py
data_preparation.py
transformer.py

## License
CTAB-GAN-Plus is licensed under Apache v2.0.

## Changes

### 1. ctabgan_synthesizer.py

_dp versions of classes and functions were created to integrate
differential privacy into the training of the model

The path of import for DataTransformer and 
ImageTransformer was adjusted to fit our directories.

### 2. data_preparation.py

The argument "test_ratio" was removed from the 
script. This argument was used to split the data into 
train and test sets. However, the test set was not kept
or used later on. Only the train set was. It also could
not be assigned to 0.