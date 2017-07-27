# ADDA-PyTorch
Pytorch implementation of Adversarial Discriminative Domain Adaptation
https://arxiv.org/abs/1702.05464

I have added batch normalization after each layer in target CNN as well as in discriminator to make the model converge for SVHN to MNIST adaptation phase, the original model was not converging for this dataset instance. 

Adding batch normalization has also improved the test accuracy by about 5% of the USPS to MNIST and MNIST to USPS adaptation process.

Feedback welcome!
