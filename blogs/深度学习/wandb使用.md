---
title: wandb炼丹伴侣
date: 2023/12/14
categories:
  - 深度学习
tags:
  - 实用工具
---

## 为什么选择wandb

![1702560177775.png](http://pic.moguw.top/i/2023/12/14/657b01b2abc85.png)

下面是wandb的重要的工具：

- Dashboard：跟踪实验，可视化结果；
- Reports：分享，保存结果；
- Sweeps：超参调优；
- Artifacts：数据集和模型的版本控制。



## 跟踪实验

### Pytorch MNIST

#### 导包

```python
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torchvision
from torchvision import transforms
import datetime
import wandb
from argparse import Namespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### Wanda config

```python
config = Namespace(
    project_name='wandb_demo',

    batch_size=512,

    hidden_layer_width=64,
    dropout_p=0.1,

    lr=1e-4,
    optim_type='Adam',

    epochs=15,
    ckpt_path='checkpoint.pt'
)
```

#### 数据加载器

```python
def create_dataloaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root="./mnist/",train=True,download=True,transform=transform)
    ds_val = torchvision.datasets.MNIST(root="./mnist/",train=False,download=True,transform=transform)

    ds_train_sub = torch.utils.data.Subset(ds_train, indices=range(0, len(ds_train), 5))
    dl_train =  torch.utils.data.DataLoader(ds_train_sub, batch_size=config.batch_size, shuffle=True,
                                            num_workers=2,drop_last=True)
    dl_val =  torch.utils.data.DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, 
                                          num_workers=2,drop_last=True)
    return dl_train,dl_val
```

#### 构建模型

```python
class CustomNet(nn.Module):
    def __init__(self, config):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=config.hidden_layer_width, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=config.hidden_layer_width,
                               out_channels=config.hidden_layer_width, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=config.dropout_p)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(config.hidden_layer_width, config.hidden_layer_width)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_layer_width, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

#### 训练函数

```python
def train_epoch(model, dl_train, optimizer):
    model.train()
    for step, batch in enumerate(dl_train):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        preds = model(features)
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model
```

#### 测试函数

```python
def eval_epoch(model, dl_val):
    model.eval()
    accurate = 0
    num_elems = 0
    for batch in dl_val:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(features)
        predictions = preds.argmax(dim=-1)
        accurate_preds = (predictions == labels)
        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()

    val_acc = accurate.item() / num_elems
    return val_acc
```

```python
def train(config=config):
    dl_train, dl_val = create_dataloaders(config)
    model = CustomNet(config).to(device)
    optimizer = torch.optim.__dict__[config.optim_type](params=model.parameters(), lr=config.lr)
    # ======================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
    model.run_id = wandb.run.id
    # ======================================================================
    model.best_metric = -1.0
    for epoch in range(1, config.epochs + 1):
        model = train_epoch(model, dl_train, optimizer)
        val_acc = eval_epoch(model, dl_val)
        if val_acc > model.best_metric:
            model.best_metric = val_acc
            torch.save(model.state_dict(), config.ckpt_path)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"epoch【{epoch}】@{nowtime} --> val_acc= {100 * val_acc:.2f}%")
        # ======================================================================
        wandb.log({'epoch': epoch, 'val_acc': val_acc, 'best_val_acc': model.best_metric})
        # ======================================================================
    # ======================================================================
    wandb.finish()
    # ======================================================================
    return model

model = train(config)
```

## 版本控制

```python
#resume the run 
import wandb 

run = wandb.init(project='wandb_demo', id= model.run_id, resume='must')
```

```python
# save dataset 
arti_dataset = wandb.Artifact('mnist', type='dataset')
arti_dataset.add_dir('mnist/')
wandb.log_artifact(arti_dataset)
```

```python
# save code 

arti_code = wandb.Artifact('ipynb', type='code')
arti_code.add_file('./mnist.ipynb')
wandb.log_artifact(arti_code)
```

```python
# save model

arti_model = wandb.Artifact('cnn', type='model')
arti_model.add_file(config.ckpt_path)
wandb.log_artifact(arti_model)
```

```python
wandb.finish() #finish时会提交保存
```



## Case分析

```python
#resume the run 
import wandb 
run = wandb.init(project=config.project_name, id= model.run_id, resume='must')
```

```python
import matplotlib.pyplot as plt 

transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root="./mnist/",train=True,download=True,transform=transform)
ds_val = torchvision.datasets.MNIST(root="./mnist/",train=False,download=True,transform=transform)
    
# visual the  prediction
device = None
for p in model.parameters():
    device = p.device
    break

plt.figure(figsize=(8,8)) 
for i in range(9):
    img,label = ds_val[i]
    tensor = img.to(device)
    y_pred = torch.argmax(model(tensor[None,...])) 
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("y_pred = %d"%y_pred)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()    
```

```python
def data2fig(data):
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(data)
    ax.set_xticks([])
    ax.set_yticks([]) 
    return fig

def fig2img(fig):
    import io,PIL
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img
```

```python
from tqdm import tqdm 
good_cases = wandb.Table(columns = ['Image','GroundTruth','Prediction'])
bad_cases = wandb.Table(columns = ['Image','GroundTruth','Prediction'])
```

```python
# 找到50个good cases 和 50 个bad cases

plt.close()

for i in tqdm(range(1000)):
    features,label = ds_val[i]
    tensor = features.to(device)
    y_pred = torch.argmax(model(tensor[None,...])) 
    
    # log badcase
    if y_pred!=label:
        if len(bad_cases.data)<50:
            data = features.permute(1,2,0).numpy()
            input_img = wandb.Image(fig2img(data2fig(data)))
            bad_cases.add_data(input_img,label,y_pred)
            
    # log goodcase
    else:
        if len(good_cases.data)<50:
            data = features.permute(1,2,0).numpy()
            input_img = wandb.Image(fig2img(data2fig(data)))
            good_cases.add_data(input_img,label,y_pred)
```

```python
wandb.log({'good_cases':good_cases,'bad_cases':bad_cases})
```

```python
wandb.finish()
```

## Sweep可视化自动调参

### 配置 Sweep config

选择一个调优算法

Sweep支持如下3种调优算法:

(1)网格搜索：grid. 遍历所有可能得超参组合，只在超参空间不大的时候使用，否则会非常慢。

(2)随机搜索：random. 每个超参数都选择一个随机值，非常有效，一般情况下建议使用。

(3)贝叶斯搜索：bayes. 创建一个概率模型估计不同超参数组合的效果，采样有更高概率提升优化目标的超参数组合。对连续型的超参数特别有效，但扩展到非常高维度的超参数时效果不好。

```python
sweep_config = {
    'method': 'random'  # grid bayes
    }
```

### 定义调优目标

设置优化指标，以及优化方向。

sweep agents 通过 wandb.log 的形式向 sweep controller 传递优化目标的值。

```python
metric = {
    'name': 'val_acc',
    'goal': 'maximize'   
    }
sweep_config['metric'] = metric
```

### 定义超参空间

超参空间可以分成 固定型，离散型和连续型。

- 固定型：指定 value
- 离散型：指定 values，列出全部候选取值。
- 连续性：需要指定 分布类型 distribution, 和范围 min, max。用于 random 或者 bayes采样。

```python
sweep_config['parameters'] = {}

# 固定不变的超参
sweep_config['parameters'].update({
    'project_name':{'value':'wandb_demo'},
    'epochs': {'value': 10},
    'ckpt_path': {'value':'checkpoint.pt'}})

# 离散型分布超参
sweep_config['parameters'].update({
    'optim_type': {
        'values': ['Adam', 'SGD','AdamW']
        },
    'hidden_layer_width': {
        'values': [16,32,48,64,80,96,112,128]
        }
    })

# 连续型分布超参
sweep_config['parameters'].update({
    
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
      },
    
    'batch_size': {
        'distribution': 'q_uniform',
        'q': 8,
        'min': 32,
        'max': 256,
      },
    
    'dropout_p': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.6,
      }
})
```

### 定义剪枝策略 (可选)

```python
sweep_config['early_terminate'] = {
    'type':'hyperband',
    'min_iter':3,
    'eta':2,
    's':3
} #在step=3, 6, 12 时考虑是否剪枝
```

```python
from pprint import pprint
pprint(sweep_config)
```

### 初始化 sweep controller 

```python
sweep_id = wandb.sweep(sweep_config, project=config.project_name)
```

### 启动 Sweep agent 

```python
# 前置略
# 该agent 随机搜索 尝试5次
wandb.agent(sweep_id, train, count=5)
```



## 调参可视化和跟踪 

### 平行坐标系图

![W&B Chart 2023_12_15 02 16 14.png](http://pic.moguw.top/i/2023/12/15/657b46ab702f3.png)

### 超参数重要性图

![1702577819364.png](http://pic.moguw.top/i/2023/12/15/657b46aa203a1.png)

## 其他案例

### PyTorch MNIST2

#### 导包

```python
import wandb
import math
import random
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
from tqdm.notebook import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

#### 数据加载器

```python
def get_dataloader(is_train, batch_size, slice=5, num_workers=2):
    """
    Get a training or testing dataloader for the MNIST dataset.
    
    """
    dataset = torchvision.datasets.MNIST(
        root=".",
        train=is_train,
        transform=T.ToTensor(),
        download=True
    )
    
    if is_train:
        shuffle = True
    else:
        shuffle = False

    subset_indices = range(0, len(dataset), slice)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    loader = torch.utils.data.DataLoader(
        dataset=subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )

    return loader
```

#### 模型搭建

```python
class SimpleModel(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10, dropout=0.0):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(dropout=0.0):
    """
    Create a simple neural network model.

    """
    model = SimpleModel(dropout=dropout).to(device)
    return model
```

#### 验证模型

```python
# 评估模型在验证数据集上的性能
def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()    # 将模型切换到评估模式，这会影响到一些具有不同行为的层，如 BatchNorm 和 Dropout
    val_loss = 0.
    with torch.inference_mode():    # 进入推断模式，这意味着模型将不会进行梯度计算，这可以提高前向传播的效率
        correct = 0
        for i, (images, labels) in tqdm(enumerate(valid_dl), leave=False):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)
            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images: # 降低数据量和减少日志文件大小
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

#### wandb记录

```python
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", # 表示图像列，用于存储图像数据
                                 "pred", # 表示预测列，用于存储模型的预测
                                 "target"] # 表示目标列，用于存储实际标签
                        +[f"score_{i}" for i in range(10)]) # 表示包含类别分数的列
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)    # 不立即提交日志，允许在记录多个日志后一次性提交，以提高效率
```

#### 训练

```python
# Launch 5 experiments, trying different dropout rates

for i in range(5):  # 启动五个不同的实验，每个实验具有不同的 dropout 率
    # 🐝 initialise a wandb run
    wandb.init(
        project="demo",
        name="pytorch_example"+str(i),
        config={
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80)})

    # Copy your config
    config = wandb.config

    # Get the data
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # A simple MLP model
    model = get_model(config.dropout)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

   # Training
    example_ct = 0
    step_ct = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(images)
            metrics = {"train/train_loss": train_loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                       "train/example_ct": example_ct}

            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb
                wandb.log(metrics)
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # If you had a test set, this is how you could log it as a Summary metric
    wandb.summary['test_accuracy'] = 0.8

    # 🐝 Close your wandb run
    wandb.finish()
```

### 总结学习的点

#### wanda记录的数据

```python
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", # 表示图像列，用于存储图像数据
                                 "pred", # 表示预测列，用于存储模型的预测
                                 "target"] # 表示目标列，用于存储实际标签
                        +[f"score_{i}" for i in range(10)]) # 表示包含类别分数的列
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)    # 不立即提交日志，允许在记录多个日志后一次性提交，以提高效率
```

#### 验证函数的构建

```python
# 评估模型在验证数据集上的性能
def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()    # 将模型切换到评估模式，这会影响到一些具有不同行为的层，如 BatchNorm 和 Dropout
    val_loss = 0.
    with torch.inference_mode():    # 进入推断模式，这意味着模型将不会进行梯度计算，这可以提高前向传播的效率
        correct = 0
        for i, (images, labels) in tqdm(enumerate(valid_dl), leave=False):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)
            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images: # 降低数据量和减少日志文件大小
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

## Pytorch CIFAR10

### 导包

```python
from __future__ import print_function
import argparse
import random # to set the python random seed
import numpy # to set the numpy random seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb
```

### 构建模型

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # In our constructor, we define our neural network architecture that we'll use in the forward pass.
        # Conv2d() adds a convolution layer that generates 2 dimensional feature maps to learn different aspects of our image
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Linear(x,y) creates dense, fully connected layers with x inputs and y outputs
        # Linear layers simply output the dot product of our inputs and weights.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Here we feed the feature maps from the convolutional layers into a max_pool2d layer.
        # The max_pool2d layer reduces the size of the image representation our convolutional layers learnt,
        # and in doing so it reduces the number of parameters and computations the network needs to perform.
        # Finally we apply the relu activation function which gives us max(0, max_pool2d_output)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Reshapes x into size (-1, 16 * 5 * 5) so we can feed the convolution layer outputs into our fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        
        # We apply the relu activation function and dropout to the output of our fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Finally we apply the softmax function to squash the probabilities of each class (0-9) and ensure they add to 1.
        return F.log_softmax(x, dim=1)
```

```python
def train(config, model, device, train_loader, optimizer, epoch):
# Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    
    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 20:
          break
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)
        
        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()
        
        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        output = model(data)
        
        # Define our loss function, and compute the loss
        loss = F.nll_loss(output, target)
        
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        
        # Update the neural network weights
        optimizer.step()
```

```python
def test(args, model, device, test_loader, classes):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)
            
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            output = model(data)
            
            # Compute the loss sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(classes[pred[0].item()], classes[target[0]])))
    
    # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})
```

### wandb配置

```python
# WandB – Initialize a new run
wandb.init(project="demo")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 4          # input batch size for training (default: 64)
config.test_batch_size = 10    # input batch size for testing (default: 1000)
config.epochs = 50             # number of epochs to train (default: 10)
config.lr = 0.1               # learning rate (default: 0.01)
config.momentum = 0.1          # SGD momentum (default: 0.5) 
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
```

### 主函数

```python
def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset: We're training our CNN on CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
    # First we define the tranformations to apply to our images
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Now we load our training and test datasets and apply the transformations defined above
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform), batch_size=config.batch_size,
                                              shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform), batch_size=config.test_batch_size,
                                             shuffle=False, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=config.momentum)
    
    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader, classes)
        
    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')

if __name__ == '__main__':
    main()
```

