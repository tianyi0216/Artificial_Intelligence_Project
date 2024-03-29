# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        # layer 1: Convultion 2d layer, output 6 channel, 1 stride with kernel size 5
        # Relu and do max2pool with kernel size 2
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, bias = True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # layer2: same as conv1 except output channel is 16
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, bias = True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # layer3, flatten to change to linear layer
        self.flat = nn.Flatten()

        # calculate the dimension of final output
        height_ly1 = (input_shape[0] - 5 + 1)//2
        height_ly2 = (height_ly1 - 5 + 1)//2
        width_ly1 = (input_shape[1] - 5 + 1) // 2
        width_ly2 = (width_ly1 -5 +1) // 2

        # layer 4, linear with output 256, follow by relu
        self.lin1 = nn.Linear(16 * height_ly2 * width_ly2, 256, bias = True) #TODO: check actual dimension
        self.relu3 = nn.ReLU()

        # layer5 , linear with output 128 and relu output
        self.lin2 = nn.Linear(256, 128, bias = True)
        self.relu4 = nn.ReLU()

        # layer 6, output 100 (or what every how many class there are)
        self.lin3 = nn.Linear(128, num_classes, bias = True)


    def forward(self, x):
        shape_dict = {} #N,C,W,H, shape dict
        # certain operations
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        #stage 1, execute layer 1, same for all following layer
        shape_dict[1] = list(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        #stage 2
        shape_dict[2] = list(x.shape)

        x = self.flat(x)
        #stage 3
        shape_dict[3] = list(x.shape)

        x = self.lin1(x)
        x = self.relu3(x)
        #stage 4
        shape_dict[4] = list(x.shape)

        x = self.lin2(x)
        x = self.relu4(x)
        # stage 5
        shape_dict[5] = list(x.shape)

        x = self.lin3(x)
        # stage 6
        shape_dict[6] = list(x.shape)
        out = x # output layer
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    x = model.named_parameters() # get all parameters using the function
    x_list = list(x) # turn into a list
    param_dict = {} # dict to store the map to layer and their shape
    for tup in x_list:
        param_dict[tup[0]] = list(tup[1].shape)
    # loop across dict and multiply the dimension to find total param
    for key in param_dict:
        # old cold a bit redundant, optimized
       # if key.split('.')[-1] == "bias":
        #    model_params += param_dict[key][0]
        #else:
        product = 1
        for num in param_dict[key]:
            product *= num
        model_params += product
    return model_params / 1e6 # turns output into millions


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
   # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################
      #  input , target = input.to(device) , target.to(device)
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
   # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for input, target in test_loader:
            #input, target = input.to(device), target.to(device)
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))