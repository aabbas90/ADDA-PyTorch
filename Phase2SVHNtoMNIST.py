from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.cuda as tcuda
import torchvision.utils as tutils
import pdb
import random
import numpy as np
import itertools
from dataset_usps import *
import math

batchSize = 256
learningRate = 2.5e-4
dSteps = 1  # To train D more
numIterations = 1000
weightDecay = 2.5e-5
betas = (0.5, 0.999)
numberHiddenUnitsD = 500
rgb2grayWeights = [0.2989, 0.5870, 0.1140]

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20 , 5)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.bn3  = nn.BatchNorm1d(500)
		
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        return out

class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        return self.fc2(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(500, numberHiddenUnitsD)
        self.fc2 = nn.Linear(numberHiddenUnitsD, numberHiddenUnitsD)
        self.fc3 = nn.Linear(numberHiddenUnitsD, 2)
        self.bn1 = nn.BatchNorm1d(numberHiddenUnitsD)
        self.bn2 = nn.BatchNorm1d(numberHiddenUnitsD)
        
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.fc1(x)))
        out = F.leaky_relu(self.bn2(self.fc2(out)))
        return self.fc3(out)

sourceTrainDataset = datasets.SVHN(root='./data/', split='train', transform=transforms.Compose([transforms.Scale(28),
                                                            transforms.ToTensor()]),
                              download=True)

sourceTestDataset = datasets.SVHN(root='./data/', split='test', transform=transforms.Compose([transforms.Scale(28),
                                                            transforms.ToTensor()]),
                              download=True)

sourceTrainLoader = torch.utils.data.DataLoader(dataset=sourceTrainDataset, batch_size=batchSize, shuffle=True)
sourceTestLoader = torch.utils.data.DataLoader(dataset=sourceTestDataset, batch_size=batchSize, shuffle=False)


#targetTrainDataset = USPSSAMPLE(root='./data/', num_training_samples=7438, train=True, transform=transforms.ToTensor())

#targetTestDataset = USPSSAMPLE(root='./data/', num_training_samples=1860, train=False, transform=transforms.ToTensor())

#targetTrainLoader = torch.utils.data.DataLoader(dataset=targetTrainDataset, batch_size=batchSize, shuffle=True)
#targetTestLoader = torch.utils.data.DataLoader(dataset=targetTestDataset, batch_size=batchSize, shuffle=False)

targetTrainDataset = datasets.MNIST(root='./data/', train=True,
                                     transform=transforms.Compose([transforms.Scale(28), transforms.ToTensor()]),
                                     download=True)

targetTestDataset = datasets.MNIST(root='./data/', train=False,
                                    transform=transforms.Compose([transforms.Scale(28), transforms.ToTensor()]),
                                    download=True)

# ,transforms.Normalize((0.1307,), (0.3081,))
# Data Loader (Input Pipeline)
targetTrainLoader = torch.utils.data.DataLoader(dataset=targetTrainDataset, batch_size=batchSize, shuffle=True)
targetTestLoader = torch.utils.data.DataLoader(dataset=targetTestDataset, batch_size=batchSize, shuffle=False)

sourceCNN = LeNet()
if ~tcuda.is_available():
    sourceCNN = torch.load('preTrainedCNNSVHN',map_location=lambda storage, loc: storage)
else:
    sourceCNN = torch.load('preTrainedCNNSVHN')

sourceCNN.eval()

targetCNN = LeNet()
if ~tcuda.is_available():
    targetCNN = torch.load('preTrainedCNNSVHN',map_location=lambda storage, loc: storage)
else:
    targetCNN = torch.load('preTrainedCNNSVHN')

targetCNN.train()

classifier = LeNetClassifier()
if ~tcuda.is_available():
    classifier = torch.load('preTrainedClassiferSVHN',map_location=lambda storage, loc: storage)
else:
    classifier = torch.load('preTrainedClassiferSVHN')

classifier.eval()

if tcuda.is_available():
    sourceCNN.cuda()
    targetCNN.cuda()
    classifier.cuda()

correct = 0
total = 0
# for images, labels in sourceTestLoader:
#     if tcuda.is_available():
#         images, labels = images.cuda(), labels.cuda()
#
#     labels = labels.long()
#     labels[torch.eq(labels, 10)] = 0
#     if images.size(1) == 3:
#         images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
#                        rgb2grayWeights[2] * images[:, 2, :, :]
#         images.unsqueeze_(1)
#
#     images = Variable(images)
#     targetOutput = targetCNN(images)
#     outputs = classifier(targetOutput)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     labels.squeeze_()
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the source test images: %d %%' % (100 * correct / total))
#
# correct = 0
# total = 0
# for images, labels in targetTestLoader:
#     if tcuda.is_available():
#         images, labels = images.cuda(), labels.cuda()
#
#     labels = labels.long()
#     labels[torch.eq(labels, 10)] = 0
#     if images.size(1) == 3:
#         images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
#                        rgb2grayWeights[2] * images[:, 2, :, :]
#         images.unsqueeze_(1)
#
#     images = Variable(images)
#     outputs = classifier(targetCNN(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the target test images: %d %%' % (100 * correct / total))

for param in sourceCNN.parameters():
    param.requires_grad = False

sourceCNN.eval()
D = Discriminator()
D.train()
targetCNN.train()

Doptimizor = optim.Adam(D.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
TargetOptimizor = optim.Adam(targetCNN.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
criteria = torch.nn.CrossEntropyLoss()

# Following Labels are in reference of D:
sourceLabels = torch.zeros(batchSize, 1).long().squeeze()
targetLabels = torch.ones(batchSize, 1).long().squeeze()

if tcuda.is_available():
    D.cuda()
    targetCNN.cuda()
    sourceCNN.cuda()
    targetLabels = targetLabels.cuda()
    sourceLabels = sourceLabels.cuda()
    criteria.cuda()

i = 0
maxTargetAcc = 60
numValidation = 500
numEpochs = int(math.ceil(float(numIterations) / float(min(len(sourceTrainLoader), len(targetTrainLoader)))))
for currentEpoch in range(numEpochs):
    targetError = 0
    DError = 0
    for it, ((sourceImages, _), (targetImages, _)) in enumerate(itertools.izip(sourceTrainLoader, targetTrainLoader)):

        if sourceImages.size(0) != targetImages.size(0):
            continue

        if tcuda.is_available():
            sourceImages = sourceImages.cuda()
            targetImages = targetImages.cuda()

        if sourceImages.size(1) == 3:
            sourceImages = rgb2grayWeights[0] * sourceImages[:,0,:,:] + rgb2grayWeights[1] * sourceImages[:,1,:,:] + rgb2grayWeights[2] * sourceImages[:,2,:,:]
            sourceImages.unsqueeze_(1)

        if targetImages.size(1) == 3:
            targetImages = rgb2grayWeights[0] * targetImages[:,0,:,:] + rgb2grayWeights[1] * targetImages[:,1,:,:] + rgb2grayWeights[2] * targetImages[:,2,:,:]
            targetImages.unsqueeze_(1)

        # Training D:
        D.zero_grad()

        sourceFeaturesForD = sourceCNN(Variable(sourceImages))
        targetFeaturesForD = targetCNN(Variable(targetImages))

        predictionOnSourceImagesForD = D(sourceFeaturesForD.detach())
        predictionOnTargetImagesForD = D(targetFeaturesForD.detach())
        predictionOnD = torch.cat((predictionOnSourceImagesForD, predictionOnTargetImagesForD), 0)
        labelsForD = torch.cat((sourceLabels, targetLabels), 0)

        DError = criteria(predictionOnD, Variable(labelsForD))
        DError.backward()

        Doptimizor.step()

        D.zero_grad()

        # Training Target:
        targetCNN.zero_grad()

        targetFeatures = targetCNN(Variable(targetImages))
        predictionOnTargetImages = D(targetFeatures)

        targetLabelsT = Variable(1 - targetLabels)

        TargetTargetError = criteria(predictionOnTargetImages, targetLabelsT)
        TargetTargetError.backward()

        if (i > 5):
            TargetOptimizor.step()
        
        targetCNN.zero_grad()

        targetError = TargetTargetError
        i = i + 1

        if (i-1) % 100 == 0:
            print('Train Itr: {} \t D Loss: {:.6f} \t Target Loss: {:.6f} \n '.format(
            i, DError.data[0], targetError.data[0]))

            if (i - 1) % 100 == 0:

                correctT = 0
                totalT = 0
                correctD = 0
                totalD = 0
				j = 0
                for images, labelsTest in targetTestLoader:
                    if tcuda.is_available():
                        images, labelsTest= images.cuda(), labelsTest.cuda()

                    labelsTest = labelsTest.long()
                    labelsTest[torch.eq(labelsTest, 10)] = 0
                    if images.size(1) == 3:
                        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                                 rgb2grayWeights[2] * images[:, 2, :, :]
                        images.unsqueeze_(1)

                    images = Variable(images)
                    outputs = classifier(targetCNN(images))
                    _, predicted = torch.max(outputs.data, 1)

                    totalT += labelsTest.size(0)
                    correctT += (predicted == labelsTest).sum()

                    _, predictedD = torch.max(outputs.data, 1)
                    totalD += predictedD.size(0)
                    labelsT = torch.ones(predictedD.size()).long()
                    if tcuda.is_available():
                        labelsT = labelsT.cuda()

                    correctD += (predictedD == labelsT).sum()
					j += 1
					if j > numValidation:
						break;

                currentAcc = 100 * correctT / totalT

                if currentAcc > maxTargetAcc:
                    torch.save(targetCNN, 'targetTrainedModel')
                    maxTargetAcc = currentAcc

                print('\n\nAccuracy of target on target test images: %d %%' % (100 * correctT / totalT))
                
				j = 0
				for images, labelsTest in sourceTestLoader:
                    if tcuda.is_available():
                        images, labelsTest = images.cuda(), labelsTest.cuda()

                    labelsTest = labelsTest.long()
                    labelsTest[torch.eq(labelsTest, 10)] = 0

                    if images.size(1) == 3:
                        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                                 rgb2grayWeights[2] * images[:, 2, :, :]
                        images.unsqueeze_(1)

                    labelsTest.squeeze_()
                    images = Variable(images)
                    outputsDFromSource = D(sourceCNN(images))

                    _, predictedD = torch.max(outputsDFromSource.data, 1)
                    totalD += predictedD.size(0)
                    labelsT = torch.zeros(predictedD.size()).long()
                    if tcuda.is_available():
                        labelsT = labelsT.cuda()

                    correctD += (predictedD == labelsT).sum()
					j += 1
					if j > numValidation:
						break;

                print('Accuracy of D on validation images: %d %%' % (100 * correctD / totalD))

# Save the Trained Model
torch.save(targetCNN, 'targetTrainedModel')
print('Max target accuracy achieved is %d %%' %maxTargetAcc)
targetCNN.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in sourceTestLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    labels.squeeze_()
    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the source test images: %d %%' % (100 * correct / total))

correct = 0
total = 0
for images, labels in targetTestLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the target test images: %d %%' % (100 * correct / total))

correct = 0
total = 0
for images, labels in targetTrainLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the target train images: %d %%' % (100 * correct / total))
