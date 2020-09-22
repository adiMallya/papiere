# DenseNet
**[8]** DenseNet
 Gao Huang(Cornell University) et al. "Densely Connected Convolutional Networks". 2017
[Paper](https://arxiv.org/pdf/1608.06993.pdf)


## Notes
- connects each layer to every other layer in feed-forwrd fashion
- network has L(L+1)/2 **direct** connections
- concatenation not element-wise addition(*used in resnets*); hence layer receives "**collective knowledge**" from preceding layers 
- Advantages :
    - Parameter efficiency : every layer adds only limited no. of parameters (e.g, 12 kernels/layer)
    - Implicit deep supervision : improved flow of gradients throughout the network
- Growth rate : 
    - determines no. of feature maps inside a dense block
    - each layer outputs k feature maps --> k is growth factor (we take k=32)
- Composite function of 3 operations --> B.N, ReLU and 3x3 Convs
- 1×1 Conv followed by 2×2 average pooling used as the *transition layers* between two contiguous dense blocks
- use *bottleneck* layers --> BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) ref as *DenseNet-B*
-  Initial conv layer outputs 2k feature maps(with *ImageNet*) using 7x7 filters, stride=2
- Training params( w.r.t ImageNet results) :
    - batch size=256
    - lr=0.1; decremented by 10 @ epoch 30 & 60
    - weight decay=10^-4, momentum=0.9 (no dampening)
    
## Architecture
![model](model.png)