# Inception-v3
**[4]** Inception v3
Christian Szegedy(Google Inc), et al. "Rethinking the Inception Architecture for Computer Vision". 2016
[Paper](https://arxiv.org/abs/1512.00567)


## Notes

- GoogleNet had low computational cost, so can be used when memory or computing power is limited or in big data scenarios.
- General design principles :
    - avoid bottlenecks with extreme compression("representaional bottlenecks")
    - Increasing the activations  per  tile --> disentangled features --> faster training
    - computational budget should be distributed in a balanced way b/w depth and width of the network

- Factorizing :
    - conv's with larger filter sizes tend to be computaionally expensive.
    -  factorize 5x5 filter to two 3x3 filters --> improves computational speed(5x5 is 2.78x more expensive)
    - assymetric convs -->  factorize convolutions of filter size nxn to a combination of 1xn and nx1 convolutions
-  Efficient Grid Size Reduction --> filter banks in the module were expanded to remove the representational bottleneck

- Auxillary classifiers --> used on top of last 17x17 layer (during training); Le et all said they act as regularizers

### Inception-v3
- Factorized 7x7 convs
- Label-smoothing regularization(LSR) --> prevents overfitting & model being confident 
- RMSProp optimizer
- momentum = 0.9, lr = 0.045
-  learning rate decay used.
- gradient clipping @ threshold = 0.2
- BatchNorm in auxilallry classifier

## Architecture 

![model](model.png) 