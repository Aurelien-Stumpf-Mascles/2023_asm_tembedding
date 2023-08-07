import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import math



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



def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, structure = "gaussian", channel=1):

    if structure == "gaussian":
        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous()).double()

    elif structure == "mean":
        window = torch.ones(size = (1,1,window_size,window_size)).double() * 1/window_size**2

    return window.type(torch.float32)

def ssim2(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = int(window_size // 2)
    print(pad)

    channels = 1.
    
    try:
        _, height, width = img1.shape
    except:
        _, height, width = img1.shape

    img1 = torch.unsqueeze(img1,dim=0) - torch.mean(img1)
    img2 = torch.unsqueeze(img2,dim=0) - torch.mean(img2)

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=1)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(input = img1, weight = window, padding=pad)
    mu2 = F.conv2d(input = img2, weight = window, padding=pad)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret


def structure(img1, img2, window_size=11, window=None, is_batch=False, full=False):

    pad = int(window_size // 2)
    
    if len(img1.shape) == 2 :
        height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
        img2 = torch.unsqueeze(img2,dim=0)
        img1 = torch.unsqueeze(img1,dim=0)
        img2 = torch.unsqueeze(img2,dim=0)
    elif len(img1.shape) == 3 :
        _, height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
    else :
        batch, _, height, width = img1.shape

    img1 -= torch.mean(img1)
    img2 -= torch.mean(img2)

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=1)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(input = img1, weight = window, padding=pad)
    mu2 = F.conv2d(input = img2, weight = window, padding=pad)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad) - mu12

    # Some constants for stability  
    C3 = (0.03 ) ** 2 / 2

    numerator =  sigma12 + C3
    denominator = np.sqrt(sigma1_sq*sigma2_sq) + C3 

    struct_score = numerator / denominator

    if is_batch:
        ret = struct_score.mean(1).mean(1).mean(1)
    else: 
        ret = struct_score.mean()
    
    return ret

def structure_batch(img1, img2, window_size=11, window=None, full=False):

    #requires that img1 and img2 are normalized
    pad = int(window_size // 2)
    
    if len(img1.shape) == 2 :
        height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
        img2 = torch.unsqueeze(img2,dim=1)
        img1 = torch.unsqueeze(img1,dim=0)
    elif len(img1.shape) == 3 :
        _, height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
    else :
        batch, _, height, width = img1.shape

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=1)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(input = img1, weight = window, padding=pad)
    mu2 = F.conv2d(input = img2, weight = window, padding=pad)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad) - mu12

    # Some constants for stability  
    C3 = (0.03 ) ** 2 / 2

    numerator =  sigma12 + C3
    denominator = np.sqrt(sigma1_sq*sigma2_sq) + C3 

    struct_score = numerator / denominator

    ret = struct_score.mean(1).mean(1).mean(1)
    
    return ret


def stable_distance(data,threshold):
    res = torch.zeros((data.shape[0],data.shape[1]))
    for session in range(data.shape[0]):
        print(session)
        for t in range(data.shape[1]):
            accu = 0
            while structure(data[session,t],data[session,t+accu]) > threshold : 
                accu += 1
                if accu + t == 464 :
                    break
            res[session,t] = accu
    return res



def generate_matrix_distance(data,distance = "euclidean"):
    """
    Returns : 
        The distance matrix of the multi-session dataset containing the correlation matrices
        distance_matrix[session1,t1,session2,t2] = metric(data[session1,t1] - data[session2,t2])
    """
    def SSIM(x,y):
        return 1 - ssim(x, y, data_range=2)

    def euclidean_metric(x,y):
        return np.linalg.norm(x - y)
    
    def exponential_metric(x,y):
        return np.exp(np.linalg.norm(x - y)/10)
    
    def STRUCTURE(x,y):
        return 1 - structure(x,y,val_range = 2)
    
    if distance == "euclidean":
        metric = euclidean_metric
    elif distance == "exp":
        metric = exponential_metric
    elif distance == "ssim":
        metric = SSIM
    elif distance == "structure":
        metric = STRUCTURE

    if len(data.shape) == 4 : 
        time,_,_,_ = data.shape
        distance_matrix = np.zeros((time,time)) 
        for t1 in range(time):
            print(t1)
            for t2 in range(t1,time):
                accu = metric(data[session1,t1,:,:,:],data[session2,t2,:,:,:])
                distance_matrix[t1,t2] = accu
                distance_matrix[t2,t1] = accu
    else : 
        nb_session,time,_,_,_ = data.shape
        distance_matrix = np.zeros((nb_session,time,nb_session,time)) 
        for session1 in range(nb_session):
            print(session1)
            for session2 in range(session1,nb_session):
                print(session2)
                for t1 in range(time):
                    print(t1)
                    for t2 in range(time):
                        accu = metric(data[session1,t1,:,:,:],data[session2,t2,:,:,:])
                        distance_matrix[session1,t1,session2,t2] = accu
                        distance_matrix[session2,t2,session1,t1] = accu

    return distance_matrix

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
    normalize = True


