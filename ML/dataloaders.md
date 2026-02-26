---
layout: page
title: How to use Datasets and Dataloaders
---





# Datasets

Dataset class is needed to packaging data. It is loaded with

```python
from torch.utils.dat import Dataset
```

There exist predetermined datasets in various libraries. For instance in ```torchvision``` or ```torchaudio```. They can be accessed via

```python
from torchvision import datasets 
```

These libraries also provide convenient methods, like ```ToTensor()```

```python
from torchvision.transforms import ToTensor
```

Let's illustrate with a complete working example how to create an object with type ```Dataset``` from the existing dataset ```MNIST```

```python
from torch.utils.dat import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root='where to store it folder', train=True, download=True, transform=ToTensor())
```




# Dataloaders

Dataloader allows quck acces to data stored in the Dataset
```python
from torch.utils.dat import Dataloader
```

Given an object ```train_data``` of Dataset type, we can create

```python
train_dataloader = Dataloader(train_data,batch_size=64,shuffle=True)
```

To access the date in the dataloader we can iterate over its elements with

```python
data, labels = next(iter(train_dataloader))
```

Calling
```python
print(data.shape)
print(labels.shape)
```

results in

<span style="color: grey;">Out:</span>
```text
torch.Size([64,train_data.shape])
torch.Size([64])
```

# Custom Datasets

See [PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) for how to construct custom Datasets



