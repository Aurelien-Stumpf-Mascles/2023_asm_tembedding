import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(in_features=input_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=4)
        self.fc3 = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x 

class SimpleDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.from_numpy(X).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.int64) - 1 #F.one_hot(torch.from_numpy(y-1).type(dtype=torch.int64), num_classes=4)

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
        while self.count + self.batch_size < len(self.dataset):
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
def Train(net,train_loader,test_loader,nb_epochs):
    net.train()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.max(1, keepdim=True)[1]
            train_correct += pred.eq(targets.view_as(pred)).sum().item()
            train_loss += loss

            compteur += 1

        if epoch % 10 == 0: 
            print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
            train_loss / (compteur * batch_size), 100 * train_correct / (compteur * batch_size)))

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
