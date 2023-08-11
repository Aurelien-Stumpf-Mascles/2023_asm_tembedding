import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearLayer(nn.Module):
    def __init__(self,input_dim,output_dim = 6):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(in_features=input_dim, out_features=1)
    def forward(self,x):
        x = self.fc1(x)
        return x 

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim = 6):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(in_features=input_dim, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=output_dim)
        self.fc6 = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc6(x)
        return x 

class MLP2(nn.Module):
    def __init__(self,input_dim,output_dim = 2):
        super(MLP2, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(in_features=input_dim, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=output_dim)
        self.fc7 = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc7(x)
        return x 
    
class CNN1(nn.Module):
    def __init__(self,n_classes):
        super(CNN1, self).__init__()
        self.fc1 = nn.Conv2d(1, 5, kernel_size=(5,5))
        self.fc2 = nn.Conv2d(5, 5, kernel_size=(3,3))
        self.fc3 = nn.MaxPool2d(kernel_size=(5, 5))
        self.fc4 = nn.Flatten()
        self.fc5 = nn.Linear(1125, 50)
        self.fc6 = nn.Linear(50,n_classes)
        self.fc7 = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        return x 

class SimpleDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y):
        """from sklearn.linear_model import LinearRegression
import numpy as np
 
# Assume you have independent variables X and a dependent variable y
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([1, 2, 3, 4])
 
# Create an instance of the LinearRegression class
reg = LinearRegression()
 
# Fit the model to the data
reg.fit(X, y)
 
# Print the coefficients of the model
print(reg.coef_)g): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.from_numpy(X).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.int64) #F.one_hot(torch.from_numpy(y-1).type(dtype=torch.int64), num_classes=4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch = self.X[idx]
        targets = self.y[idx]

        return batch,targets



class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = torch.utils.data.DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
# === Train === ###
def Train(net,train_loader,test_loader,nb_epochs,lr,criterion = nn.NLLLoss()):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    batch_size = train_loader.batch_sampler.n_samples * train_loader.batch_sampler.n_classes

    # train loop
    for epoch in range(nb_epochs):
        train_correct = 0
        train_loss = 0
        compteur = 0
        
        # loop per epoch 
        for i, (batch, targets) in enumerate(train_loader):

            output = net(batch)

            loss = criterion(output, targets)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.max(1, keepdim=True)[1]
            train_correct += pred.eq(targets.view_as(pred)).sum().item()
            train_loss += loss

            compteur += 1

        if epoch % 3 == 0 : 
            print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
            train_loss / compteur, 100 * train_correct / (compteur * batch_size)))

            if test_loader is not None :
                
                # loop, over whole test set
                compteur_batch = 0
                net.eval()
                test_correct = 0

                for i, (batch, targets) in enumerate(test_loader):
                    
                    output = net(batch)
                    pred = output.max(1, keepdim=True)[1]
                    test_correct += pred.eq(targets.view_as(pred)).sum().item()
                    compteur_batch+=1
                    
                print('Test accuracy {:.2f}%'.format(
                    100 * test_correct / (test_loader.batch_size * compteur_batch)))
                
                net.train()
            
    print('End of training.\n')
