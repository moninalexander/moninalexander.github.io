---
layout: page
title: "MNIST: Convolutional NN"
---


## Importing libraries

Importing necessary libraries

```python
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary # For summary
from torchvision import datasets
from torchvision.transforms import ToTensor # For converting images to tensors
import numpy as np
import matplotlib.pyplot as plt
```

## Preparing Data

Data structure for MNIST: training set is 60000 tuples, test set is 10000 tuples. 

Each tuple is a $$28\times 28$$ tensor and  1 of 100 labels, i.e. $$(\text{img}(1\times32\times32), \text{label})$$

### Importing Data

```python
train_data=datasets.MNIST(root='../data',
            download=True,train=True, transform=ToTensor())
test_data=datasets.MNIST(root='../data',
            download=True,train=False, transform=ToTensor())
```

#### Global Parameters

```python
NUM_CLASSES = len(train_data.classes)
TRAIN_SIZE = len(train_data)
TEST_SIZE = len(test_data)
X, _ = train_data[0]
IM_CH = X.shape[0]
IM_HIHGT = X.shape[1]
IM_WIDTH = X.shape[2]
BATCH_SIZE = 64
```

### Creating Dataloaders

```python
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

print(train_loader.dataset.data.shape)
```

<span style="color: grey;">Out:</span>
```text
torch.Size([60000, 28, 28])
```

## Plotting examples


```python
_, axes = plt.subplots(8,8,figsize=(6,6))
X, y = next(iter(train_loader))
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')
    # ax.set_title(y[i].item())
    ax.imshow(X[i].squeeze(),cmap='bone')
plt.show()
```
![MNIST Architecture]({{ "../assets/images/sample.png" | relative_url }})



## Models

### Fully connected


See details in [Building blocks...](/ML/building_blocks/)

```python
class FC_MNIST_2(nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.box = nn.Sequential(
            nn.Linear(IM_HIHGT*IM_WIDTH, n1),
            nn.ReLU(),
            nn.Linear(n1 ,n2),
            nn.ReLU(),
            nn.Linear(n2, NUM_CLASSES)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.box(x)
        return logits
```

Creating an instance

```python
model_fc = FC_MNIST_2(512,512)
summary(model_fc,depth=3,input_size=(64,28,28))
```


<span style="color: grey;">Out:</span>
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
FC_MNIST_2                               [64, 10]                  --
├─Flatten: 1-1                           [64, 784]                 --
├─Sequential: 1-2                        [64, 10]                  --
│    └─Linear: 2-1                       [64, 512]                 401,920
│    └─ReLU: 2-2                         [64, 512]                 --
│    └─Linear: 2-3                       [64, 512]                 262,656
│    └─ReLU: 2-4                         [64, 512]                 --
│    └─Linear: 2-5                       [64, 10]                  5,130
==========================================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 42.86
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 0.53
Params size (MB): 2.68
Estimated Total Size (MB): 3.41
==========================================================================================
```


## Convolutional


```python
class Conv_MNIST_21(nn.Module):
    def __init__(self, ch1, ks1, padding1, pks1, ch2, ks2, padding2, pks2):
        super().__init__()
        self.p1 = padding1
        self.p2 = padding2
        self.ks1 = ks1
        self.ks2 = ks2
        self.pks1 = pks1
        self.pks2 = pks2
        self.ch2 = ch2
        self.last = int(((IM_HIHGT+2*self.p1+1-self.ks1)/self.pks1+2*self.p2+1-self.ks2)/self.pks2)**2*ch2
        self.box = nn.Sequential(
            nn.Conv2d(IM_CH, ch1, kernel_size=(ks1, ks1), padding=padding1),
            nn.ReLU(),
            nn.MaxPool2d(pks1),
            nn.Conv2d(ch1, ch2, kernel_size=(ks2, ks2), padding=padding2),
            nn.ReLU(),
            nn.MaxPool2d(pks2),
            nn.Flatten(),
            nn.Linear(self.last,NUM_CLASSES)
        )
    def forward(self, x):
        x = self.box(x)
        return x
```

with an instant

```python
model_conv=Conv_MNIST_21(32,3,1,2,32,3,1,2)
summary(model_conv,depth=3,input_size=(64,1,28,28))
```

<span style="color: grey;">Out:</span>
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Conv_MNIST_21                            [64, 10]                  --
├─Sequential: 1-1                        [64, 10]                  --
│    └─Conv2d: 2-1                       [64, 32, 28, 28]          320
│    └─ReLU: 2-2                         [64, 32, 28, 28]          --
│    └─MaxPool2d: 2-3                    [64, 32, 14, 14]          --
│    └─Conv2d: 2-4                       [64, 32, 14, 14]          9,248
│    └─ReLU: 2-5                         [64, 32, 14, 14]          --
│    └─MaxPool2d: 2-6                    [64, 32, 7, 7]            --
│    └─Flatten: 2-7                      [64, 1568]                --
│    └─Linear: 2-8                       [64, 10]                  15,690
==========================================================================================
Total params: 25,258
Trainable params: 25,258
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 133.07
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 16.06
Params size (MB): 0.10
Estimated Total Size (MB): 16.36
==========================================================================================
```

## Training



```python
def train_run(data_loader, model, loss_fn, optimizer):
    model.train()
    batch_loss = []
    correct = 0
    total = 0 
    for batch, (X, y) in enumerate(data_loader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.item())
        correct+=(torch.argmax(y_pred,1)==y).sum().item()
        total+=len(y)
    return sum(batch_loss)/len(batch_loss), correct/total


def test_run(data_loader, model, loss_fn, optimizer):
    model.eval()
    batch_loss = []
    correct = 0
    total = 0 
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            batch_loss.append(loss.item())
            correct+=(torch.argmax(y_pred,1)==y).sum().item()
            total+=len(y)
    return sum(batch_loss)/len(batch_loss), correct/total
```


```python
LEARNING_RATE = 0.001
J = nn.CrossEntropyLoss()
optimizer_fc = torch.optim.Adam(model_fc.parameters(),lr=LEARNING_RATE)
optimizer_conv = torch.optim.Adam(model_conv.parameters(),lr=LEARNING_RATE)

scheduler_fc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fc, mode="max",patience=5, threshold=0.0001)
scheduler_conv = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode="max",patience=5, threshold=0.0001)
```



```python
test_acc_fc = []
test_loss_fc= []
test_acc_conv = []
test_loss_conv = []

train_acc_fc = []
train_loss_fc= []
train_acc_conv = []
train_loss_conv = []
```


```python
epochs = 20
print("=" * 70)
print(f"{'Epoch':<10}{'Model':<10}{'Test Accuracy':<20}{'Test Loss':<15}{'Learning Rate':<15}")
for t in range(epochs):
    train_loss, train_acc=train_run(train_loader, model_fc, J, optimizer_fc)
    test_loss, test_acc=test_run(test_loader, model_fc, J, optimizer_fc)
    scheduler_fc.step(test_acc)
    
    print("=" * 70)
    print(f"{t:<10}{'FC':<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler_fc.get_last_lr()[-1]:<15.6f}")

    test_acc_fc.append(test_acc)
    test_loss_fc.append(test_loss)
    train_acc_fc.append(train_acc)
    train_loss_fc.append(train_loss)
    
    
    ##################################################################
    
    train_loss, train_acc = train_run(train_loader, model_conv, J, optimizer_conv)
    test_loss, test_acc=test_run(test_loader, model_conv, J, optimizer_conv)
    scheduler_conv.step(test_acc)

    print("-" * 70)
    print(f"{'':<10}{'CNN':<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler_fc.get_last_lr()[-1]:<15.6f}")
    
    test_acc_conv.append(test_acc)
    test_loss_conv.append(test_loss)
    
    train_acc_conv.append(train_acc)
    train_loss_conv.append(train_loss)
```


<span style="color: grey;">Out:</span>
```text
============================================================
Epoch          Test Accuracy  Test Loss      Learning Rate  
============================================================
0              0.9648         0.1203         0.001000       
```








```python

```


```python

```


```python

```


```python

```


```python

```














```python

```


```python

```


```python

```


```python

```


```python

```



<span style="color: grey;">Out:</span>
```text

```







<span style="color: grey;">Out:</span>
```text

```