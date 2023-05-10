import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=transform)
    test_set=datasets.FashionMNIST('./data', train=False,transform=transform)

    loader = torch.utils.data.DataLoader(train_set, batch_size = 64) if training else torch.utils.data.DataLoader(test_set, batch_size = 64)

    return loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(28**2,128), nn.ReLU(), nn.Linear(128, 64),nn.ReLU(), nn.Linear(64, 10))
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # optimizer
    model.train() # set model before training

    # train for T times
    for epoch in range(T):
        running_loss = 0.0 # track for total loss
        total = 0 # batch sizes
        accurate = 0 # accurate prediction number
        for i, data in enumerate(train_loader, 0):
            inputs, label = data # input and label
            opt.zero_grad()
            
            outputs = model(inputs) # output from model
            
            # get max predictions and update data by check for accuracy
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)
            total += label.size(0)
            accurate += (predicted == label).sum().item()
            loss.backward()
            opt.step() # Gradient descent
            running_loss += loss.item() * label.size(0)
        
        # calculate information and print the information
        running_loss /= total
        accuracy = accurate/total
        print(f"Train Epoch: {epoch} Accuracy: {accurate}/{total}({accuracy*100:.2f}%) Loss: {running_loss:.3f}")

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    # set to eval model
    model.eval()
    accurate = 0
    total = 0
    loss_result = 0
    with torch.no_grad():
        # loop through test data and compare with label
        for data, labels in test_loader:
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()
            loss_result += loss.item() * labels.size(0)

    # calcualte needed information and print
    accuracy = accurate/total
    loss_result /= total
    if show_loss:
        print(f'Average loss: {loss_result:.4f}')
        print(f'Accuracy: {accuracy*100:.2f}%')
    else:
        print(f'Accuracy: {accuracy*100:.2f}%')

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    # provided class names
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    # get output and convert to probability using softmax
    outputs = model(test_images[index])
    prob = F.softmax(outputs, dim = 1)

    # find the highest 3 and print out the information
    idx_max = []
    
    for idx in range(3):
        max_idx = None
        for i in range(len(prob[0])):
            if not i in idx_max:
                if max_idx == None or prob[0][i] > prob[0][max_idx]:
                    max_idx = i
        idx_max.append(max_idx)

    idx1 = idx_max[0]
    idx2 = idx_max[1]
    idx3 = idx_max[2]
    predict_1 = class_names[idx1]
    predict_2 = class_names[idx2]
    predict_3 = class_names[idx3]

    prob_1 = prob[0][idx1]
    prob_2 = prob[0][idx2]
    prob_3 = prob[0][idx3]

    print(f'{predict_1}: {prob_1*100:.2f}%')
    print(f'{predict_2}: {prob_2*100:.2f}%')
    print(f'{predict_3}: {prob_3*100:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # code for testing
    criterion = nn.CrossEntropyLoss() #- commented out for debugging- need to add back
    train_loader = get_data_loader()
    #print(type(train_loader))
    #print(train_loader.dataset)
    test_loader = get_data_loader(False)
    #print(type(test_loader))
    #print(test_loader.dataset)
    model = build_model()
    #print(model)
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss=False)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)