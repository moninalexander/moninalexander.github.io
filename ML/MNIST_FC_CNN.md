---
layout: page
title: "MNIST: Fully connected vs Convolutional"
---

## Intro

The `MNIST` dataset is sufficiently simple and can be considered solved, in the sense that relatively simple architectures achieve very high accuracy, outperforming humans, whose accuracy is about [`97.5-98%`](https://en.wikipedia.org/wiki/MNIST_database).Working with this dataset is therefore primarily educational. 

In this note, I compare the performance of fully connected neural networks `FC` and convolutional neural networks `CNN` on the `MNIST` dataset. I deliberately restrict attention to simple models and avoid data augmentation in order to isolate the effect of the architecture.

This note is not intended as a deep dive into Machine Learning `ML`, but rather as an introductory hands-on project to become familiar with `PyTorch`, `tensors`, model construction, and training. The presentation follows a notebook style with minimal commentary. See [https://github.com/moninalexander/dl-lightning](https://github.com/moninalexander/dl-lightning) for more detail.



## Importing libraries

For this project, we need to import the following libraries

```python
import torch
from torchvision import datasets # Getting the data
from torchvision.transforms import ToTensor, Normalize, Compose # For converting images to tensors
from torch.utils.data import DataLoader # Providing the data in batches
import matplotlib.pyplot as plt # For plotting
from torch import nn # Necessary building blocks for our models
from torchinfo import summary # For creating models' summaries
import time
```

`datasets` allows us to download and manage the `MNIST` dataset. The MNIST training set contains 60000 samples, and the test set contains 10000 samples. Each sample is a tuple of the form `(img, label)`, where `img` is a 28 by 28 grayscale (one-channel) image, and label is an integer in `{0,...,9}`.

`ToTensor`, `Normalize`, `Compose` are used to convert the images to `PyTorch` tensors and rescale them using the global mean `0.1307` and standard deviation `0.3081`, computed from the training set.

`DataLoader` splits the dataset into batches, shuffles them if needed, and provides them during training and evaluation.





## Preparing the data

First we download the dataset and store it locally. If the dataset has already been downloaded, the local copy will be used automatically.


```python
img_transform = Compose([ToTensor(), Normalize(0.1307,0.3081)])
train_data=datasets.MNIST(
    root='../data',
    download=True,
    train=True, 
    transform=img_transform
)
test_data=datasets.MNIST(
    root='../data',
    download=True,
    train=False, 
    transform=img_transform
)
```

Now that the data is ready, we extract a few global parameters that will be used throughout the note:

```python
NUM_CLASSES = len(train_data.classes)
TRAIN_SIZE = len(train_data)
TEST_SIZE = len(test_data)
X, _ = train_data[0]
IM_CH = X.shape[0]
IM_HIGHT = X.shape[1]
IM_WIDTH = X.shape[2]
BATCH_SIZE = 64
```

We use the `DataLoader` to split both the training and test datasets into batches of size `BATCH_SIZE = 64`

```python
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
print(train_loader.dataset.data.shape)
```

<span style="color: grey;">Out:</span>
```text
torch.Size([60000, 28, 28])
```

Finally, we plot several example images from the dataset. For more details on plotting with `matplot.pyplot` see a short note on [Pyplot](/ML/pyplot/)


```python
_, axes = plt.subplots(8,8,figsize=(6,6))
X, y = next(iter(train_loader))
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')
    # ax.set_title(y[i].item())
    ax.imshow(X[i].squeeze(),cmap='bone')
plt.show()
```
![MNIST_EXAMPLES]({{ "../assets/images/sample.png" | relative_url }})



## Models

The fully connected neural network has two hidden layers with `n1` and `n2` neurons, respectively

```python
class FC_MNIST_2(nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.box = nn.Sequential(
            nn.Linear(IM_HIGHT*IM_WIDTH, n1),
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

The convolutional neural network combines two `nn.Conv2d` layers and two `nn.MaxPool2d` layers, followed by a single fully connected layer

```python
class Conv_MNIST_21(nn.Module):
    def __init__(self, ch1, ch2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(IM_CH, ch1, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch1, ch2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            X = torch.randn(1, IM_CH, IM_HIGHT, IM_WIDTH)
            self.last_numel = self.feature_extractor(X).numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last_numel,NUM_CLASSES)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
```

Even though we could vary the filter size, we keep it fixed at $$3 \times 3$$. Similarly, the
`MaxPooling` layers use filters of size $$2 \times 2$$. We only vary the number of channels in the two convolutional layers.

Note that we could compute the output size explicitly using the formula that relates the input size, kernel sizes, padding, and pooling. Instead, we determine the required number of features dynamically by passing a random tensor of the same size through the `feature_extractor` and calling `numel()`.



For more details on building models, see [Building blocks...](/ML/building_blocks/)



## Training

Now we come to training. For every model, we will be providing data in batches of `BATCH_SIZE = 64`, computing gradients and backpropagating.

When computing the accuracy, we take into account that the default `reduction` for the `CrossEntropyLoss` function is `mean`, meaning that the loss is averaged over the batch. Since not all batches have the same size, the average over batches is not equal to the average over all samples. Since we need the latter, we multiply the `CrossEntropyLoss` by the batch size and accumulate the result.


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
        batch_loss.append((len(y)*loss).item())
        correct+=(torch.argmax(y_pred,1)==y).sum().item()
        total+=len(y)
    return sum(batch_loss)/TRAIN_SIZE, correct/total


def test_run(data_loader, model, loss_fn):
    model.eval()
    batch_loss = []
    correct = 0
    total = 0 
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            batch_loss.append((len(y)*loss).item())
            correct+=(torch.argmax(y_pred,1)==y).sum().item()
            total+=len(y)
    return sum(batch_loss)/TEST_SIZE, correct/total
```
The test accuracy is quantized in units of `0.0001`, since the test set contains `10000` samples. We therefore take a `threshold` for improvement smaller than this value.



```python
LEARNING_RATE = 0.001
J = nn.CrossEntropyLoss()
optimizer_fc = torch.optim.Adam(model_fc.parameters(),lr=LEARNING_RATE)
scheduler_fc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fc, mode="max",patience=5, factor=0.5, threshold=0.0)
```

To keep track of the `train` and `test` accuracies and losses we store them in separate python lists. For that, we define the accuracy and loss histories


```python
test_acc_fc = []
test_loss_fc= []
train_acc_fc = []
train_loss_fc = []
```

and initiate training with


```python
epochs = 20
print("=" * 70)
print(f"{'Epoch':<10}{'Model':<10}{'Test Accuracy':<20}{'Test Loss':<15}{'LR':<15}")
for t in range(epochs):
    train_loss, train_acc=train_run(train_loader, model_fc, J, optimizer_fc)
    test_loss, test_acc=test_run(test_loader, model_fc, J)
    scheduler_fc.step(test_acc)
    
    print("=" * 70)
    print(f"{t:<10}{'FC':<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler_fc.get_last_lr()[-1]:<15.6f}")

    test_acc_fc.append(test_acc)
    test_loss_fc.append(test_loss)
    train_acc_fc.append(train_acc)
    train_loss_fc.append(train_loss)
```


### Fully connected models

Let us examine the results for several choices of parameters.

#### Tiny: (2,2)

We deliberately start with ridiculously small numbers of neurons

```python
model_fc = FC_MNIST_2(1,1)
```

The training results are as follows

<span style="color: grey;">Out:</span>
```text
======================================================================
Epoch     Model     Test Accuracy       Test Loss      LR             
======================================================================
0         FC        0.2014              2.0219         0.001000       
======================================================================
1         FC        0.2109              1.9269         0.001000       
======================================================================
2         FC        0.2290              1.8808         0.001000       
.
.
.
======================================================================
18        FC        0.3565              1.6584         0.001000       
======================================================================
19        FC        0.3764              1.6345         0.001000       
======================================================================
Time      36.3 s     
```

The simplicity of this model is deceptive. 

Random guessing would result in an accuracy of about 10%. This model performs significantly better: it correctly classifies roughly one third of the images. The reason is that the model has a sufficiently large number of trainable parameters `807`:

```python
summary(model_fc,depth=3,input_size=(BATCH_SIZE,IM_HIGHT,IM_WIDTH))
```


<span style="color: grey;">Out:</span>
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
FC_MNIST_2                               [64, 10]                  --
├─Flatten: 1-1                           [64, 784]                 --
├─Sequential: 1-2                        [64, 10]                  --
│    └─Linear: 2-1                       [64, 1]                   785
│    └─ReLU: 2-2                         [64, 1]                   --
│    └─Linear: 2-3                       [64, 1]                   2
│    └─ReLU: 2-4                         [64, 1]                   --
│    └─Linear: 2-5                       [64, 10]                  20
==========================================================================================
Total params: 807
Trainable params: 807
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.05
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 0.01
Params size (MB): 0.00
Estimated Total Size (MB): 0.21
==========================================================================================
```



#### Unnecessary large

As the opposite extreme, we consider a model with `512` neurons in each hidden layer. This model has a whopping `670k` trainable parameters.

```python
model_fc = FC_MNIST_2(512,512)
summary(model_fc,depth=3,input_size=(BATCH_SIZE,IM_HIGHT,IM_WIDTH))
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

The accuracy virtually plateaus after around 25 epochs

<span style="color: grey;">Out:</span>
```text
======================================================================
Epoch     Model     Test Accuracy       Test Loss      LR             
======================================================================
0         FC        0.9622              0.1230         0.001000       
======================================================================
1         FC        0.9775              0.0725         0.001000       
======================================================================
.
.
.
======================================================================
27        FC        0.9869              0.1047         0.000500       
======================================================================
28        FC        0.9865              0.1053         0.000500       
======================================================================
29        FC        0.9859              0.1085         0.000500       
======================================================================
.
.
.
======================================================================
48        FC        0.9864              0.1323         0.000063       
======================================================================
49        FC        0.9863              0.1339         0.000063       
======================================================================
Time     171.6 s
```

In fact, a smaller model with around `100k` trainable parameters 


```python
model_fc = FC_MNIST_2(128,128)
summary(model_fc,depth=3,input_size=(BATCH_SIZE,IM_HIGHT,IM_WIDTH))
```

<span style="color: grey;">Out:</span>
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
FC_MNIST_2                               [64, 10]                  --
├─Flatten: 1-1                           [64, 784]                 --
├─Sequential: 1-2                        [64, 10]                  --
│    └─Linear: 2-1                       [64, 128]                 100,480
│    └─ReLU: 2-2                         [64, 128]                 --
│    └─Linear: 2-3                       [64, 128]                 16,512
│    └─ReLU: 2-4                         [64, 128]                 --
│    └─Linear: 2-5                       [64, 10]                  1,290
==========================================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 7.57
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 0.14
Params size (MB): 0.47
Estimated Total Size (MB): 0.81
==========================================================================================
```

can produce a comparable accuracy

<span style="color: grey;">Out:</span>
```text
======================================================================
Epoch     Model     Test Accuracy       Test Loss      LR             
======================================================================
0         FC        0.9543              0.1436         0.001000       
======================================================================
.
.
.
======================================================================
48        FC        0.9850              0.1380         0.000063       
======================================================================
49        FC        0.9850              0.1404         0.000063       
======================================================================
Time     109.1 s
```


#### Lesson learned

Fully connected two-layer networks can achieve strong performance on MNIST. However, their parameter count grows rapidly with the hidden layer size. Moreover, even models with hundreds of thousands of parameters plateau around `98.5-98.6%` percent accuracy without data augmentation.


![MNIST_ACC_LOSS]({{ "../assets/images/MNIST_FC.png" | relative_url }})


Now we will consider convolutional neural networks


### Convolutional neural networks

We start with a model using `(8, 8)` channels

```python
model_conv=Conv_MNIST_21(8,8)
```

The training results in

<span style="color: grey;">Out:</span>
```text
======================================================================
Epoch     Model     Test Accuracy       Test Loss      Learning Rate  
0         CNN       0.9636              0.1166         0.001000       
1         CNN       0.9764              0.0711         0.001000
.
.
.
17        CNN       0.9868              0.0450         0.000500       
18        CNN       0.9857              0.0470         0.000500       
19        CNN       0.9870              0.0452         0.000500       
======================================================================
Time     245.2 s
```


This model has only around `4.5k` trainable parameters, as shown below, an order of magnitude less than the fully connected network with a comparable accuracy.

```python
summary(model_conv,depth=3,input_size=(BATCH_SIZE,1,IM_HIGHT,IM_WIDTH))
```


<span style="color: grey;">Out:</span>
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Conv_MNIST_21                            [64, 10]                  --
├─Sequential: 1-1                        [64, 10]                  --
│    └─Conv2d: 2-1                       [64, 8, 28, 28]           80
│    └─ReLU: 2-2                         [64, 8, 28, 28]           --
│    └─MaxPool2d: 2-3                    [64, 8, 14, 14]           --
│    └─Conv2d: 2-4                       [64, 8, 14, 14]           584
│    └─ReLU: 2-5                         [64, 8, 14, 14]           --
│    └─MaxPool2d: 2-6                    [64, 8, 7, 7]             --
│    └─Flatten: 2-7                      [64, 392]                 --
│    └─Linear: 2-8                       [64, 10]                  3,930
==========================================================================================
Total params: 4,594
Trainable params: 4,594
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 11.59
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 4.02
Params size (MB): 0.02
Estimated Total Size (MB): 4.24
==========================================================================================
```




However, the convolutional neural network can do even better. Let's increase the number of channels in the second layer following the
#### <span style="color: red;">Design principle:</span>
As spatial resolution decreases due to pooling, the number of channels should increase. This compensates for the loss of spatial detail by increasing the representational capacity along the feature dimension.

For the model

```python
model_conv=Conv_MNIST_21(32,64)
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
│    └─Conv2d: 2-4                       [64, 64, 14, 14]          18,496
│    └─ReLU: 2-5                         [64, 64, 14, 14]          --
│    └─MaxPool2d: 2-6                    [64, 64, 7, 7]            --
│    └─Flatten: 2-7                      [64, 3136]                --
│    └─Linear: 2-8                       [64, 10]                  31,370
==========================================================================================
Total params: 50,186
Trainable params: 50,186
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 250.08
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 19.27
Params size (MB): 0.20
Estimated Total Size (MB): 19.67
==========================================================================================
```

we get

<span style="color: grey;">Out:</span>
```text
======================================================================
Epoch     Model     Test Accuracy       Test Loss      Learning Rate  
0         CNN       0.9836              0.0518         0.001000       
1         CNN       0.9869              0.0358         0.001000       
.
.
.  
18        CNN       0.9923              0.0456         0.000500       
19        CNN       0.9923              0.0471         0.000500       
======================================================================
Time     460.6 s
```

![MNIST_ACC_LOSS_CONV]({{ "../assets/images/MNIST_CNN.png" | relative_url }})

We can acheive an accuracy of around `99 - 99.1%` even with smaller models, but this one consistently acheives an accuracy of around `99.2 - 99.3%`

## Conclusion

Convolutional neural networks achieve higher accuracy than fully connected networks on `MNIST` while using significantly fewer parameters. Even relatively small CNNs outperform larger fully connected models. The key advantage comes from exploiting spatial structure.


| Model         |  Parameters | Accuracy | Time |
| ------------- | 
| FC(1, 1)    |              807        | ~37.6%        | 36 s     
| FC(128, 128) |          118k       | ~98.5%        | 109 s         
| FC(512, 512) |          670k       | ~98.6%        | 172 s
| CNN(8, 8) |              4.6k       | ~98.7%        | 245 s 
| CNN(32, 64) |            50k        | ~99.2%        | 461 s


