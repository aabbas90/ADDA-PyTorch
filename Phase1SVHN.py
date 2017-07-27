from __future__ import print_function
import torch
import torch.cuda as tcuda
import torchvision.utils as tutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import math

numIterations = 5000
batchSize = 128
learningRate = 0.001
weightDecay = 2.5e-4
rgb2grayWeights = [0.2989, 0.5870, 0.1140]
train_dataset = datasets.SVHN(root='./data/', split='train', transform=transforms.Compose([transforms.Scale(28),
                                                            transforms.ToTensor()]), download=True)

test_dataset = datasets.SVHN(root='./data/', split='test', transform=transforms.Compose([transforms.Scale(28),
                                                            transforms.ToTensor()]), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

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

model = LeNet()
classifier = LeNetClassifier()

optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),  lr=learningRate, weight_decay = weightDecay)

model.train()
classifier.train()
if tcuda.is_available():
    model.cuda()
    classifier.cuda()

losses = []
numEpochs = int(math.ceil(float(numIterations) / float(len(train_loader))))
for currentEpoch in range(numEpochs):
    for bid, (data, labels) in enumerate(train_loader):
        if tcuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        labels[torch.eq(labels, 10)] = 0
        labels = torch.squeeze(labels).long()
        if data.size(1) == 3:
            redChannel = rgb2grayWeights[0] * data[:,0,:,:]
            greenChannel = rgb2grayWeights[1] * data[:,1,:,:]
            blueChannel = rgb2grayWeights[2] * data[:,2,:,:]
            data = redChannel + greenChannel + blueChannel
            data.unsqueeze_(1)

        data, labels = Variable(data), Variable(labels)
        # Init
        optimizer.zero_grad()
        # Predict
        y_pred = classifier(model(data))

        # Calculate loss
        loss = F.cross_entropy(y_pred, labels)
        losses.append(loss.data[0])
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Display
    print('\n Train Epoch: {} \tLoss: {:.6f}'.format(
            currentEpoch + 1,
            loss.data[0]))
        # tutils.save_image(data.data,'trainingBatch {}.jpg'.format(batchIdx + 1))

# Test the Model
model.eval()  
correct = 0
total = 0
correctClass = torch.zeros(10)
totalClass = torch.zeros(10)
for images, labels in test_loader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels[torch.eq(labels, 10)] = 0
    labels = torch.squeeze(labels).long()
    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(model(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    for i in range(len(correctClass)):
        classInPrediction = predicted == i
        classInLabels = labels == i
        correctClass[i] += (classInPrediction * classInLabels).sum()
        totalClass[i] += (classInLabels).sum()

print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
for i in range(len(correctClass)):
    print('\nTest Accuracy of the model on the Class %d : %d %%' % (i, 100 * correctClass[i] / totalClass[i]))
# Save the Trained Model
torch.save(model, 'preTrainedCNNSVHN')
torch.save(classifier, 'preTrainedClassiferSVHN')