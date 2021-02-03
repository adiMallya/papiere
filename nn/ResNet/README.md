# ResNet
**[7]** ResNet
 Kaiming He(Microsoft Research) et al. "Deep Residual Learning for Image Recognition". 2016
[Paper](https://arxiv.org/pdf/1512.03385.pdf)


## Notes
- **Why is stacking more layers not OK ?**
    - Vanishing/Exploding gradients 
        - was addressed by normalized initialization and intermediate normalization layers
    - Degradation problem ?

- addreses the degradation by introducing *deep residual learning*
    - idea is deeper network must not produce higher training error than its shallower counterpart
    - So a deep nn can be constructed from a shallow nn by copying the weights and using *identity mappings* in the gaps
- Use skip/shortcut connections to perform *identity mapping*
    -  it is easier to optimize the residual mapping( F(x) = H(x) - x) than to optimizethe original, unreferenced mapping --> identity x will remain even if gradeints are vanishing
    -  major benefit of identity mapping --> enables backpropagation signal to reach from output layers to input layers
- connection skipping 2 layers worked best
- skip connections add neither extra parameter nor computa-tional complexity
- Linear projections for dimension matching --> basically zero-padding shorcuts for incresing dimensions; this will add up to params
- Use BatchNorm
- Mini-batch size of 256
- lr of 0.1 ; divided by 10 when error plateaus
- weight decay = 0.0001, momentum = 0.9
- replace 2-layer blocks in ResNet-34 with 3-layer *bottlenecks*, for deeper architectures with complexity still low
- ResNet-152 (11.3 billion FLOPs) still has lower complexity than VGG-16/19 nets (15.3/19.6 billion FLOPs)<sup id="a1">[1](#f1)</sup>




## Architecture 

#### ResNet-50
![model](model.png)
  




<b id="f1">1</b> FLOPS = floating point operations/sec,  to measure the computational cost, with multiplications, addi- tions and multiply-add operations considered. [↩](#a1)