#import package
# * model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary #https://pypi.org/project/torch-summary/
from torch import optim
from torch.optim.lr_scheduler import StepLR
# from resnet_torch import resnet50
from torchvision.models import resnet50

# * dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# * display images
from torchvision import utils
import matplotlib.pyplot as plt
# %matplotlib inline

# * utils
import numpy as np
from torchsummary import summary
import time
import copy 
# end --------------------------- #

# * data path
dir_path = "../data/sample/"

# if not os.path.exists(dir_path):
#     os.mkdir(dir_path)
# end --------------------------- #


# * load dataset
train_data = datasets.STL10(dir_path, split='train', download=True, 
                            transform=transforms.ToTensor())
valid_data = datasets.STL10(dir_path, split='test', download=True,
                            transform=transforms.ToTensor())
print(f"len of train_data: {len(train_data)}")
print(f"len of valid_data: {len(valid_data)}")
# end --------------------------- #


# * To normalize the dataset, calculate the mean and std
train_meanRGB = [np.mean(x.numpy(), axis=(1,2) ) for x, _ in train_data]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_data]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])

train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

valid_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in valid_data]
valid_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in valid_data]

valid_meanR = np.mean([m[0] for m in valid_meanRGB])
valid_meanG = np.mean([m[1] for m in valid_meanRGB])
valid_meanB = np.mean([m[2] for m in valid_meanRGB])

valid_stdR = np.mean([s[0] for s in valid_stdRGB])
valid_stdG = np.mean([s[1] for s in valid_stdRGB])
valid_stdB = np.mean([s[2] for s in valid_stdRGB])

print(f"train_mean. R: {train_meanR}, G: {train_meanG}, B: {train_meanB}\n")
print(f"valid_mean. R: {valid_meanR}, G: {valid_meanG}, B: {valid_meanB}\n")
# end --------------------------- #


# * define the image transformation.
train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([train_meanR, train_meanG, train_meanB],
                          [train_stdR, train_stdG, train_stdB]),
    transforms.RandomHorizontalFlip(),
])
val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([train_meanR, train_meanG, train_meanB],
                         [train_stdR, train_stdG, train_stdB]),
])
# end --------------------------- #


# * transformation을 dataset에 적용하고, dataloader를 생성합니다.
# apply transformation
train_data.transform = train_transformation
valid_data.transform = val_transformation

# create DataLoader
train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_data, batch_size=32, shuffle=True)
# end --------------------------- #


# transformation이 적용된 sample image확인
# display sample images
def show(img, y=None, color=True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr)
    
    if y is not None:
        plt.title('labels: '+str(y))
        
np.random.seed(1)
torch.manual_seed(1)
    
grid_size = 4
rnd_inds = np.random.randint(0, len(train_data), grid_size)
print(f'image indices: {rnd_inds}')

x_grid = [ train_data[i][0] for i in rnd_inds ]
y_grid = [ train_data[i][1] for i in rnd_inds ]

x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)

show(x_grid, y_grid)
# end --------------------------- #


#model test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50().to(device)
x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
print(output.size())

summary(model, (3, 224, 224), device=device.type)
# end --------------------------- #


# * model training
loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, 
                                 patience=10)


# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True) # return idx
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
    
# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item(), metric_b

# epoch당 loss를 정의
def loss_epoch(model, loss_func, dataset_dl, 
               sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
            
        if sanity_check is True:
            break
        
    loss = running_loss / len_data
    metric = running_metric / len_data
    
    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    valid_dl=params["valid_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, valid_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_wts = copy.deepcopy(model.state_dict())

            # torch.save(model.state_dict(), path2weights)
            # print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history
    
    
    
    
# define the training hyper parameters
params_train = {
    'num_epochs':1,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'valid_dl':valid_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')
createFolder('./models')


model, loss_hist, metric_hist = train_val(model, params_train)

# Train-Validation Progress
num_epochs=params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()