import os
import time
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms



dataTrans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
 
data_dir = './data'
train_data_dir = './data/train'
val_data_dir = './data/val'
train_dataset = datasets.ImageFolder(train_data_dir, dataTrans)
print(train_dataset.class_to_idx)
val_dataset = datasets.ImageFolder(val_data_dir, dataTrans)
 
image_datasets = {'train':train_dataset,'val':val_dataset}
    
 
    # wrap your data and label into Tensor
 
    
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4) for x in ['train', 'val']}
 
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
 
    # use gpu or not
use_gpu = torch.cuda.is_available()



def train_model(model, lossfunc, optimizer, scheduler, num_epochs=10):
    start_time = time.time()
 
    best_model_wts = model.state_dict()
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
 
            running_loss = 0.0
            running_corrects = 0.0
 
            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data
                
 
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
 
                # zero the parameter gradients
                optimizer.zero_grad()
 
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = lossfunc(outputs, labels)
 
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
 
                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
 
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
 
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
 
    # load best model weights
    model.load_state_dict(best_model_wts)
  
    return model
    
    
   # get model and replace the original fc layer with your fc layer
model_ft = models.resnet50(pretrained=True, progress=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
 
if use_gpu:
    model_ft = model_ft.cuda()
 
    # define loss function
lossfunc = nn.CrossEntropyLoss()
 
params = list(model_ft.fc.parameters())
optimizer_ft = optim.SGD(params, lr=0.001, momentum=0.9)
 
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
 
model_ft = train_model(model=model_ft,
                           lossfunc=lossfunc,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=50)
                           
                           
torch.save(model_ft.state_dict(), './model/model.pth', _use_new_zipfile_serialization=False)



from math import exp
import numpy as np
 
from PIL import Image
import cv2
 
 
 
infer_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
 
 
IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
LABEL_OUTPUT_KEY = 'predicted_label'
MODEL_OUTPUT_KEY = 'scores'
LABELS_FILE_NAME = 'labels.txt'
 
 
def decode_image(file_content):
    image = Image.open(file_content)
    image = image.convert('RGB')
    return image
 
 
def read_label_list(path):
    with open(path, 'r',encoding="utf8") as f:
        label_list = f.read().split(os.linesep)
    label_list = [x.strip() for x in label_list if x.strip()]
    return label_list
 
 
def resnet50(model_path):
 
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path,map_location ='cpu'))
    # model.load_state_dict(torch.load(model_path))
 
    model.eval()
 
    return model
 
 
def predict(file_name):
    LABEL_LIST = read_label_list('./model/labels.txt')
    model = resnet50('./model/model.pth')
    
    image1 = decode_image(file_name)
    
 
    input_img = infer_transformation(image1)
 
    input_img = torch.autograd.Variable(torch.unsqueeze(input_img, dim=0).float(), requires_grad=False)
 
    logits_list =  model(input_img)[0].detach().numpy().tolist()
    print(logits_list)
    maxlist=max(logits_list)
    print(maxlist)
 
    z_exp = [exp(i-maxlist) for i in  logits_list]
 
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    print(softmax)
    labels_to_logits = {
        LABEL_LIST[i]: s for i, s in enumerate(softmax)
    }
    
    predict_result = {
        LABEL_OUTPUT_KEY: max(labels_to_logits, key=labels_to_logits.get),
        MODEL_OUTPUT_KEY: labels_to_logits
    }
 
    return predict_result
 
file_name = './data/test/ASC-US&LSIL/03424.jpg'
result = predict(file_name)  #可以替换其他图片
import matplotlib.pyplot as plt
 
plt.figure(figsize=(10,10)) #设置窗口大小
img = decode_image(file_name)
plt.imshow(img)
plt.show()
 
print(result)
