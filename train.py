import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import json
import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import json
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchattacks
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets,models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import learn2learn as l2l
import random
if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")


import time


random_seed = 22
torch.manual_seed(random_seed)
# val_size = 2500

random_seed = 22
torch.manual_seed(random_seed)



train_transform = transforms.Compose([transforms.Resize(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3444, 0.3803, 0.4078], [0.2027, 0.1369, 0.1156])
                                 ])

test_transform = transforms.Compose([transforms.Resize(32),
                                 transforms.ToTensor(),
                                transforms.Normalize([0.3444, 0.3803, 0.4078], [0.2027, 0.1369, 0.1156])
                                    ])
class ApplyTransform(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if transform is None and target_transform is None:
            print("Transform is not implemented :)")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)




def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert(nb_nodes>0 and nb_nodes<=10)

    digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))
    digits2=torch.arange(3) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    digits = torch.cat((digits,digits2))

    print(digits)
   
    digits_split=list()
    i=0
    for n in range(nb_nodes, 0, -1):
        inc=int((10-i)/n)
        digits_split.append(digits[i:i+3])
        print
        i+=inc
        print(inc)
        print(digits_split)
    
    print(digits_split)
    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=nb_nodes*n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = next(dataiter)

    data_splitted=list()
    for i in range(nb_nodes):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))
    print(data_splitted)
    return data_splitted

def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)

    data_splitted=list()

    for _ in range(nb_nodes):
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(next(dataiter))), batch_size=batch_size, shuffle=shuffle))

    return data_splitted



def  get_splitData(type, n_samples_train, n_samples_test, n_clients, batch_size, shuffle,train_dataset , test_dataset):

        if type=="iid":
            train=iid_split(train_dataset, n_clients, n_samples_train, batch_size, shuffle)
            test=iid_split(test_dataset, n_clients, n_samples_test, batch_size, shuffle)
        elif type=="non_iid":
            train=non_iid_split(train_dataset, n_clients, n_samples_train, batch_size, shuffle)
            test=non_iid_split(test_dataset, n_clients, n_samples_test, batch_size, shuffle)
        else:
            train=[]
            test=[]

        return train, test



def loss_classifier(predictions,labels):
    
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    return loss(predictions ,labels)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    acc= (predictions == targets).sum().float() / targets.size(0)
    return acc


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0)).byte()
    adaptation_indices[torch.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[1 - adaptation_indices], labels[1 - adaptation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    
    return valid_error, valid_accuracy
def fast_adapt_metrics(batch, learner, loss, adaptation_steps, shots, ways, device):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0)).byte()
    adaptation_indices[torch.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[1 - adaptation_indices], labels[1 - adaptation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        # train_error /= len(adaptation_data)
        learner.adapt(train_error)
        
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    # valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    # valid_accuracy = l2l.utils.accuracy(predictions, evaluation_labels)

    return valid_error,valid_accuracy
def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0
    
    for idx,(features,labels) in enumerate(dataset):
        features = features.to(device)
        labels=labels.to(device)
        predictions= model(features)
        loss+=loss_f(predictions,labels)
    
    loss/=idx+1
    return loss


def accuracy_dataset(model, dataset):
    
    correct=0
    
    for features,labels in iter(dataset):
        features = features.to(device)
        labels=labels.to(device)
        predictions= model(features)
        _,predicted=predictions.max(1,keepdim=True)
        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
    accuracy = 100*correct/len(dataset.dataset)
    return accuracy


def train_step(model, mu:int, optimizer, train_data, loss_f,loss,meta_batch_size,
        adaptation_steps,shots,ways):
    total_loss=0
    meta_train_error = 0.0
    meta_train_accuracy = 0.0
    meta_valid_error = 0.0
    meta_valid_accuracy = 0.0
    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        learner = model.clone()
        batch=train_data.sample()

        evaluation_error,evaluation_accuracy= fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        meta_train_accuracy += evaluation_accuracy.item()
        
#     for idx, (features,labels) in enumerate(train_data):
#         optimizer.zero_grad()
#         features = features.to(device)
#         predictions= model(features)
#         labels=labels.to(device)
#         loss=loss_f(predictions,labels)
#         loss+=mu/2*difference_models_norm_2(model,model_0)
#         total_loss+=loss
#         loss.backward()
#         optimizer.step()
#     print("meta accuracy: ", meta_train_accuracy/meta_batch_size)    
    return meta_train_error/meta_batch_size

def meta_metrics(model,  optimizer, train_data, loss,meta_batch_size,
        adaptation_steps,shots,ways):
 
    losses=[]
    accuracy=[]
    for k in range(len(train_data)):
        total_loss=0
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            learner = model.clone()
            batch= train_data[k].sample()

            evaluation_error, evaluation_accuracy = fast_adapt_metrics(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()
        losses.append(meta_train_error/meta_batch_size)
        accuracy.append(meta_train_accuracy/meta_batch_size)
        print(meta_train_accuracy/meta_batch_size)
    return losses,accuracy



def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f,loss,meta_batch_size,
        adaptation_steps,shots,ways):
    
#     model_0=deepcopy(model)
    
    for e in range(epochs):
        optimizer.zero_grad()
        local_loss=train_step(model,mu,optimizer,train_data,loss_f,loss,meta_batch_size,
        adaptation_steps,shots,ways)

        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()
    return local_loss


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def average_models(model, clients_models_hist:list , weights:list):

    """Creates the new model of a given iteration with the models of the other

    clients"""
    
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model



def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, lr=0.0005, meta_lr=0.01, mu=0, 
            file_name="test", epochs=5, meta_batch_size=8, adaptation_steps=5, shots=2, ways=5):

    loss_hist=[]
    acc_hist=[]
    loss_f=loss_classifier
    model= l2l.algorithms.MetaSGD(model, lr=meta_lr)
    local_optimizer = optim.Adam(model.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])    
    losses,acc= meta_metrics(model,  local_optimizer, training_sets, loss,meta_batch_size,
        adaptation_steps,shots,ways)
    loss_hist.append(losses)
    acc_hist.append(acc)
    print(loss_hist)
 # CHANGE FOR META LEARNING
#     loss_hist=[[float(loss_dataset(model, dl, loss_f).detach()) 
#         for dl in training_sets]]
#     print(loss_hist)
    
#     acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
 # CHANGE FOR META LEARNING

    server_hist=[[tens_param.detach().cpu().numpy()
        for tens_param in list(model.parameters())]]
    models_hist = []

    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    server_loss_list=[]
    server_accuracy_list=[]  
    
    for i in range(n_iter):
        
        clients_params=[]
        clients_models=[]
        clients_losses=[]
        
        for k in range(K):

            local_model=deepcopy(model) #meta sgd model
#             local_optimizer=optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)
#             exp_lr_scheduler = lr_scheduler.StepLR(local_optimizer, step_size=7, gamma=0.1)
#             local_model= l2l.algorithms.MetaSGD(local_model, lr=meta_lr)
                
            local_optimizer = optim.Adam(local_model.parameters(),lr=lr) #meta sgd
            loss = nn.CrossEntropyLoss(reduction='mean')    
            local_loss=local_learning(local_model,mu,local_optimizer,
                training_sets[k],epochs,loss_f,loss,meta_batch_size,adaptation_steps,shots,ways)
            clients_losses.append(local_loss)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)    
            clients_models.append(deepcopy(local_model))

            print(f"{k}---local_loss--- {local_loss}" )
        
        
        model = average_models(deepcopy(model), clients_params, 
            weights=weights)
        models_hist.append(clients_models)
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        # Function TO CHANGE FOR META LEARNING
#         loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) 
#             for dl in training_sets]]
#         acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]
        losses,acc= meta_metrics(model, local_optimizer, testing_sets, loss,meta_batch_size,
        adaptation_steps)
        loss_hist.append(losses)
        acc_hist.append(acc)
        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        server_accuracy_list.append(server_acc)
        server_loss_list.append(server_loss)        

        server_hist.append([tens_param.detach().cpu().numpy() 
            for tens_param in list(model.parameters())])
    return model, loss_hist, acc_hist,server_accuracy_list,server_loss_list


def performattack(dataloader, batch_size,model):
    attack = torchattacks.PGD(model, eps=2/255, alpha=2/225)
    poisoned_images = []
    poisoned_labels = []
    for images, labels in dataloader:
        images = attack(images, labels)
        poisoned_images.append(images)
        poisoned_labels.append(labels) # labels are not changing
    
    poisoned_dataset = TensorDataset(torch.cat(poisoned_images, dim=0), torch.cat(poisoned_labels, dim=0))
    poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=False)
    
    return poisoned_dataloader




def train_attack(batch_size, poison, data_split, optimizer, comm_rounds, local_epochs, lr, num_clients, ways, meta_lr, shots,dataset_root):
    print(f"Training Configuration:")
    print(f"Batch Size: {batch_size}, Poison Level: {poison}%, Data Split: {data_split}, Optimizer: {optimizer}")
    print(f"Communication Rounds: {comm_rounds}, Local Epochs: {local_epochs}, Learning Rate: {lr}")
    print(f"Number of Clients per Round: {num_clients}, Ways: {ways}, Meta Learning Rate: {meta_lr}, Shots: {shots}")
    # Training logic will be implemented here
    # This is a placeholder for the actual training function which would include the training loop, 
    # model updates, and communication between the server and clients.
    start_time = time.time()
    train_dataset=datasets.ImageFolder(root=dataset_root)


    test_size = 2500
    train_size = len(train_dataset)  - test_size

    train_transform = transforms.Compose([transforms.Resize(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3444, 0.3803, 0.4078], [0.2027, 0.1369, 0.1156])
                                 ])

    test_transform = transforms.Compose([transforms.Resize(32),
                                 transforms.ToTensor(),
                                transforms.Normalize([0.3444, 0.3803, 0.4078], [0.2027, 0.1369, 0.1156])
                                    ])

    train_ds, test_ds  = random_split(train_dataset, [train_size, test_size])
    train_ds = ApplyTransform(train_ds, train_transform)
    test_ds = ApplyTransform(test_ds,test_transform)

    train_dls, test_dls = get_splitData(data_split,
    n_samples_train =500, n_samples_test=300, n_clients =num_clients, 
    batch_size = batch_size, shuffle =True,train_dataset=train_ds,test_dataset=test_ds)





    
    model =models.resnet18(weights='IMAGENET1K_V1')

# for param in model.parameters():
#     param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, ways)


    model = model.to(device)
    if poison > 0:
        poison_idx = random.randint(num_clients - 1, size=(poison))
        augs = []
        for i in range(len(poison_idx)):
            train_dls[i] = performattack(train_dls[i], batch_size,model)


    training_tasks = []
    testing_tasks = []

    for i in range(len(train_dls)):
        train_dataset = l2l.data.MetaDataset(train_dls[i].dataset)
        # valid_dataset = l2l.data.MetaDataset(valid_dataset)

        train_transforms = [
            l2l.data.transforms.NWays(train_dataset, ways),
            l2l.data.transforms.KShots(train_dataset, 2*shots),
            l2l.data.transforms.LoadData(train_dataset),
            l2l.data.transforms.RemapLabels(train_dataset),
            l2l.data.transforms.ConsecutiveLabels(train_dataset),
        ]
        train_tasks = l2l.data.TaskDataset(train_dataset,
                                           task_transforms=train_transforms,
                                           num_tasks=1000)
        training_tasks.append(train_tasks)

    for i in range(len(test_dls)):
        test_dataset = l2l.data.MetaDataset(test_dls[i].dataset)
        test_transforms = [
            l2l.data.transforms.NWays(test_dataset, ways),
            l2l.data.transforms.KShots(test_dataset, 2 * shots),
            l2l.data.transforms.LoadData(test_dataset),
            l2l.data.transforms.RemapLabels(test_dataset),
            l2l.data.transforms.ConsecutiveLabels(test_dataset),
        ]
        test_tasks = l2l.data.TaskDataset(test_dataset,
                                          task_transforms=test_transforms,
                                          num_tasks=600)
        testing_tasks.append(test_tasks)

    model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list = FedProx( model, training_tasks, 
                                                                                           comm_rounds, testing_tasks,
   lr=0.0005, meta_lr=0.01, epochs =50,meta_batch_size=batch_size,
        adaptation_steps=5,)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time) 




    