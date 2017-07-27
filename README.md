# ADDA-PyTorch
Pytorch implementation of Adversarial Discriminative Domain Adaptation
https://arxiv.org/abs/1702.05464

For running any experiment first run the relevant Phase1xxxx file which will train sourceCNN and will write the network weights which will then be used in the relevant Phase2xxxxtoyyyy file which is for the adaptation phase. 

I have added batch normalization after each layer in source and target CNN as well as in discriminator to make the model converge for SVHN to MNIST adaptation phase, the original model was not converging for this dataset instance. 

Adding batch normalization has also improved the test accuracy by about 5% of the USPS to MNIST and MNIST to USPS adaptation process.

Feedback welcome!
