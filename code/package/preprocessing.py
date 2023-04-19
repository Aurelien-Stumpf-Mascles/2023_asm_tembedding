import numpy as np
import torch

def create_labels_with_transitions(a,min_counts = 30):
    """
    Args :
        The initial labels of the time windows

    Returns :
        For each time window, the main label if its occurence is greater than return_counts 
        else the transition label denoted by 0
    """

    def f(x):
        unique,counts = np.unique(x,return_counts=True)
        if np.max(counts) < min_counts:
            return 0
        else : 
            idx = np.argmax(counts)
            return unique[idx]
        
    return np.apply_along_axis(arr=a,func1d=f,axis=2)
    
def generate_corr_matrix(data):
    """
    Returns :
        Generate the correlation coefficient matrix for each session and time window in the dataset
        Requires the knowledge of single
    """

    if len(data.shape) == 2 : 
        n,m,_ = data.shape
        corr_matrix = np.zeros((n,m,m))
        for i in range(n):
            corr_matrix[i,:,:] = np.corrcoef(data[i,:,:])
        corr_matrix = torch.unsqueeze(torch.from_numpy(corr_matrix),dim=2)
    else :
        n,m,p,_ = data.shape
        corr_matrix = np.zeros((n,m,p,p))
        for i in range(n):
            for j in range(m):
                corr_matrix[i,j,:,:] = np.corrcoef(data[i,j,:,:])
        corr_matrix = torch.unsqueeze(torch.from_numpy(corr_matrix),dim=2)
    return corr_matrix

def generate_matrix_distance(data):
    """
    Returns : 
        The distance matrix of the multi-session dataset containing the correlation matrices
        distance_matrix[session1,t1,session2,t2] = metric(data[session1,t1] - data[session2,t2])
    """
    if len(data.shape) == 2 : 
        time,_,_,_ = data.shape
        distance_matrix = np.zeros((time,time)) 
        for t1 in range(time):
            for t2 in range(time):
                distance_matrix[t1,t2] = np.linalg.norm(data[t1,0,:,:] - data[t2,0,:,:])
    else : 
        nb_session,time,_,_ = data.shape
        distance_matrix = np.zeros((nb_session,time,nb_session,time)) 
        for session1 in range(nb_session):
            print(session1)
            for t1 in range(time):
                for session2 in range(nb_session):
                    for t2 in range(time):
                        if session1 == session2 : 
                            distance_matrix[session1,t1,session2,t2] = 0
                        else :
                            distance_matrix[session1,t1,session2,t2] = np.linalg.norm(data[session1,t1,0,:,:] - data[session2,t2,0,:,:])
    return distance_matrix

def flatten_higher_triangular(data):
    """
    Returns :
        Generate the flattened higher triangular of the correlation coefficient matrix for each session 
        and time window in the dataset
    """
    if len(data.shape) == 2 : 
        n,m,_, = data.shape
        res = np.zeros((n,m*(m-1)//2))
        for i in range(n):
            accu = torch.Tensor([])
            for k in range(m-1):
                accu = torch.cat([accu,data[i,k,k+1:]])
            res[i,:] = accu
    else :
        n,m,p,_ = data.shape
        res = np.zeros((n,m,p*(p-1)//2))
        for i in range(n):
            for j in range(m):
                accu = torch.Tensor([])
                for k in range(p-1):
                    accu = torch.cat([accu,data[i,j,k,k+1:]])
                res[i,j,:] = accu
    return res

def generate_vector_distance(data,distance = "euclidean"):
    """
    Returns : 
        The distance matrix of the multi-session dataset containing the flattened higher triangular of the data
        distance_matrix[session1,t1,session2,t2] = metric(data[session1,t1] - data[session2,t2])
    """

    def euclidean_metric(x,y):
        return np.linalg.norm(x - y)
    
    def exponential_metric(x,y):
        return np.exp(np.linalg.norm(x - y)/10)
    
    if distance == "euclidean":
        metric = euclidean_metric
    if distance == "exp":
        metric = exponential_metric

    if len(data.shape) == 2 : 
        time,_ = data.shape
        distance_matrix = np.zeros((time,time)) 
        for t1 in range(time):
            for t2 in range(t1,time):
                accu = metric(data[t1,:],data[t2,:])
                distance_matrix[t1,t2] = accu
                distance_matrix[t2,t1] = accu
    else : 
        nb_session,time,_ = data.shape
        distance_matrix = np.zeros((nb_session,time,nb_session,time)) 
        for session1 in range(nb_session):
            print(session1)
            for t1 in range(time):
                for session2 in range(session1,nb_session):
                    for t2 in range(time):
                        accu = metric(data[session1,t1,:],data[session2,t2,:])
                        distance_matrix[session1,t1,session2,t2] = accu
                        distance_matrix[session2,t2,session1,t1] = accu
    return distance_matrix