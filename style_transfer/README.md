# Neural Style Transfer
**[4]** Style transfer
Leon A. Gatys, et al. "Image Style Transfer Using Convoutional Neural Networks". 2016
[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)


## Notes
- Algorithm tires to separate and recombine the image content and style of natural images
- Use VGG-19
- Content
    - feature responses in higher layers of NN represent *content*
    - content for target image is output from conv4_2
    - **Content loss** : mean squared distances b/w representaions in content & target images
- Style
    - including correlations b/w multiple layers you obtain multiscale *style* representation of image
    - this is calculated at layers conv1_1 - conv5_1
    - correlations are given by : <mark>*Gram matrix*</mark> that contains non-localised info
    - **Style loss** : squared distances b/w gram matrices of style & target images, along with style weights 'w'
- While adding them up, constant weights alpha and beta are added to the losses
- smaller <mark>alpha/beta</mark> ratio --> more the stylistic effect