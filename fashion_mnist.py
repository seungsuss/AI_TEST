import torch 
import torch.nn as nn 
from torchvision import datasets , transforms
from torch.utils.data import random_split,DataLoader,Subset
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
import os
from PIL import Image

dataset_down_path = "/Users/seungsulee/Documents/dataset"
model_save_path = "model_save/mnist_model.pth"

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Anke boot'] ##10가지 분류 레이블

def check_device(str):
    if str == "mac" :
        device="mps" if torch.backends.mps.is_available() else "cpu"
    else :
        device="cuda" if torch.cuda.is_available() else "cpu"

    return device

def mnist_init(show_dataset =  0):
    print("---------------------------------------------")
    print("--            FASHION MNIST MODEL          --")
    print("--                    program date: 2024.9 --")
    print("--                         program by odin --")
    print("---------------------------------------------")

    device = check_device("mac")    

    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ]) 

    traindf=datasets.FashionMNIST(root=dataset_down_path,download=True,train=True,transform=transform)
    testdf=datasets.FashionMNIST(root=dataset_down_path,download=True,train=False,transform=transform)

    traindf[0][0].max(),traindf[0][0].min()    

    idx=np.random.randint(0,len(traindf),25)
    sample=Subset(traindf,idx)
    test_img = testdf[4][0]
    label_classes=traindf.classes

    traindf,valdf=random_split(traindf,[len(traindf)-10000,10000])
    len(traindf),len(valdf)

    trainloader=DataLoader(traindf,64,shuffle=True)
    testloader=DataLoader(testdf,64,shuffle=True)
    valloader=DataLoader(valdf,64,shuffle=True)

    #tensor_PIL = transforms.ToPILImage()
    #image = tensor_PIL(test_img)
    #image.save("test2.png")
    #plt.figure(figsize=(1,1))
    #plt.imshow(image,cmap="gray")
    #plt.show()   
    

    if show_dataset == 1:
        plt.figure(figsize=(10,8))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.imshow(sample[i][0].squeeze(),cmap="gray")
            plt.title(label_classes[sample[i][1]])
            plt.axis("off")
    
        plt.tight_layout()
        plt.show()

    return device , trainloader, testloader , valloader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()        
        
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.LeakyReLU(),        # LeakyReLU activation function after the first layer
            
            # Second Linear layer: input is 128 features, output is 64 features
            nn.Linear(128, 64),
            nn.LeakyReLU(),        # LeakyReLU activation function after the second layer
            
            # Third Linear layer: input is 64 features, output is 10 classes (for FashionMNIST)
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # Flatten the input tensor to be of shape (batch_size, 28*28)
        x = x.view(-1, 28*28)
        
        # Pass the flattened input through the defined model layers
        return self.model(x)

def inference(mymodel, image_file_path):

    image = Image.open(image_file_path)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)

    device , trainloader, testloader , valloader = mnist_init()

    mymodel.to(device)
    img_tensor.to(device)

    #### input image debug
    #tensor_PIL = transforms.ToPILImage()
    #image = tensor_PIL(img_tensor)
    #image.save("test.png")
    #plt.figure(figsize=(1,1))
    #plt.imshow(image,cmap="gray")
    #plt.show()

    mps_tensor = img_tensor.to(device)

    pred = mymodel(mps_tensor)
    cpu_pred = pred.to('cpu')    
    num_pred = cpu_pred.detach().numpy()
    #num_pred = np.array(cpu_pred)
    search_idx = num_pred.argmax()
    
    return class_names[search_idx]

def train(model, trainloader, loss_fn, optimizer, metric, device):
    # Initialize a list to store the loss for each batch
    total_loss = []
    
    # Set the model to training mode (important for layers like dropout or batch norm)
    model.train()
    
    # Loop over each batch of images and labels from the DataLoader
    for images, labels in trainloader:
        # Move the images and labels to the specified device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear the gradients from the previous step to avoid accumulation
        optimizer.zero_grad()
        
        # Forward pass: Get model predictions for the current batch of images
        preds = model(images)
        
        # Calculate the loss between the predictions and the actual labels
        loss = loss_fn(preds, labels)
        
        # Backward pass: Compute gradients for the model's parameters
        loss.backward()
        
        # Update the model's parameters using the optimizer
        optimizer.step()
        
        # Record the loss for this batch
        total_loss.append(loss.item())

        metric.update(preds, labels)
        
    # Calculate the average loss over all batches
    avg_loss = np.mean(total_loss)
    
    # Compute the overall accuracy for the entire epoch
    avg_acc = metric.compute().item()
    
    # Reset the metric state for the next epoch
    metric.reset()
    
    # Return the average loss and accuracy, rounded to four decimal places
    return round(avg_loss, 4), round(100 * avg_acc, 4)

def validate(model, vailloader, loss_fn, metric, device):
    total_loss = []
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculations
        for images, labels in vailloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get predictions
            preds = model(images)
            
            # Calculate loss between predictions and true labels
            lss = loss_fn(preds, labels)
            total_loss.append(lss.item()) 
            
            # Update the metric with current batch predictions and labels
            metric.update(preds, labels)
    
    # Calculate the average loss over all batches
    avg_loss = np.mean(total_loss)
    
    # Compute the overall accuracy
    avg_acc = metric.compute().item()
    
    metric.reset()
    return round(avg_loss, 4), round(100 * avg_acc, 4)

def set_loss_func(model, device):
    loss=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    metric=Accuracy(task="multiclass",num_classes=10).to(device)

    return loss, opt, metric

def mnist_model_proc(mode , epoch_num = 0 , input_file_path = ""):

    device , trainloader, testloader , valloader = mnist_init(show_dataset = 0)

    mymodel=Model()
    mymodel.to(device)
    mymodel(torch.rand(10,28,28).to(device)).shape

    #loss=nn.CrossEntropyLoss()
    #opt=torch.optim.Adam(mymodel.parameters(),lr=0.001)
    #metric=Accuracy(task="multiclass",num_classes=10).to(device)
   
    loss, opt, metric = set_loss_func(mymodel, device)

    if os.path.isfile(model_save_path):
        mymodel = torch.load(model_save_path)

    if mode == 0 : ## train
        train(mymodel, trainloader, loss, opt, metric, device) # check
        validate(mymodel, valloader, loss, metric, device)    
        
        for i in range(epoch_num):
            avg_loss_train, avg_acc_train = train(mymodel, trainloader, loss, opt, metric, device)
            avg_loss_test, avg_acc_test = validate(mymodel, valloader, loss, metric, device)
    
            print(f"For epoch {i+1}, the training loss is {avg_loss_train} and accuracy is {avg_acc_train}%, "
              f"and for validation, loss is {avg_loss_test} and accuracy is {avg_acc_test}%.")
        torch.save(mymodel,model_save_path) 

    elif mode == 1 : ## inference
        result_str = inference(mymodel, input_file_path) 

        print("Inference Result : ", result_str)

        

if __name__ == '__main__':

    device , trainloader, testloader , valloader, test_tensor = mnist_init()

    mymodel=Model()
    #mymodel.to(device)
    mymodel.to('mps')
    mymodel(torch.rand(10,28,28).to(device)).shape

    #loss=nn.CrossEntropyLoss()
    #opt=torch.optim.Adam(mymodel.parameters(),lr=0.001)
    #metric=Accuracy(task="multiclass",num_classes=10).to(device)
   
    loss, opt, metric = set_loss_func(mymodel, device)

    if os.path.isfile(model_save_path):
        mymodel = torch.load(model_save_path)

    mymodel.to('mps')
    tensor_PIL = transforms.ToPILImage()
    image = tensor_PIL(test_tensor)
    image.save("test1.png")
    plt.figure(figsize=(1,1))
    plt.imshow(image,cmap="gray")
    plt.show()

    mps_tensor = test_tensor.to(device)
    pred = inference(mymodel, mps_tensor)
    cpu_pred = pred.to('cpu')
    print(cpu_pred)
    num_pred = cpu_pred.detach().numpy()
    #num_pred = np.array(cpu_pred)
    search_idx = num_pred.argmax()
    print(class_names[search_idx])    
    

    train(mymodel, trainloader, loss, opt, metric, device) # check
    validate(mymodel, valloader, loss, metric, device)    

    num_epochs = 10
    for i in range(num_epochs):
        avg_loss_train, avg_acc_train = train(mymodel, trainloader, loss, opt, metric, device)
        avg_loss_test, avg_acc_test = validate(mymodel, valloader, loss, metric, device)
    
        print(f"For epoch {i+1}, the training loss is {avg_loss_train} and accuracy is {avg_acc_train}%, "
              f"and for validation, loss is {avg_loss_test} and accuracy is {avg_acc_test}%.")
        

    torch.save(mymodel,model_save_path)
    



    


    
    
    