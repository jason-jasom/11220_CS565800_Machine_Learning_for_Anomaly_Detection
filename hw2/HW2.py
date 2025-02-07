import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.fft import fft,ifft
from sklearn.metrics import roc_auc_score,pairwise_distances

np.random.seed(0)

def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    Resample the data to balance classes.

    Parameters:
        data: np.array, shape=(n_samples, n_features)
            Input data.
        label: np.array, shape=(n_samples,)
            Labels corresponding to the data samples.
        outlier_ratio: float, optional (default=0.01)
            Ratio of outliers to include in the resampled data.
        target_label: int, optional (default=0)
            The label to be treated as normal.

    Returns:
        new_data: np.array
            Resampled data.
        new_label: np.array
            Resampled labels.
    """
    new_data = []
    new_label = []
    for i in [1, -1]:
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * outlier_ratio
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False
            )
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * 1)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * 0)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

def sample_by_label(data, label, number=10, target_label=0):
    new_data = []
    new_label = []
    for i in [0,1]:
        if i == target_label:
            i_data = data[label == i]
            idx = np.random.choice(
                list(range(len(i_data))), min(number,len(i_data)), replace=False
            )
            new_data.append(i_data[idx])
    return new_data

#problem 1    
def visualization(train_data, train_label,test_data, test_label,category):
    draw_normal_1 = sample_by_label(train_data, train_label, number=10, target_label=0)
    draw_anomaly_1 = sample_by_label(test_data, test_label, number=10, target_label=1)
    fig_1, axs_1 = plt.subplots(2,1)
    plt.suptitle(f'{category} Dataset')
    axs_1[0].set_title("Anomaly Sample")
    for i in draw_anomaly_1[0]:
        axs_1[0].plot(list(range(len(draw_anomaly_1[0][0]))),i,color='red')
    axs_1[1].set_title("Normal Sample")
    for i in draw_normal_1[0]:
        axs_1[1].plot(list(range(len(draw_normal_1[0][0]))),i,color='blue')
    plt.tight_layout()

#problem 2
def knn(train_data,test_data,test_label,k):
    k_near_distance_list = []
    all_data = np.concatenate((train_data,test_data))
    distance_mat = pairwise_distances(all_data,metric='euclidean')
    for id,test in enumerate(test_data):
        distance_list = distance_mat[len(train_data)+id,:len(train_data)].copy()
        distance_list.sort()
        k_near_distance_list.append(np.mean(distance_list[:k]))
    return roc_auc_score(test_label,k_near_distance_list)

#problem 3
def PCA_(train_data, train_label,test_data, test_label,k = 5,N = 5):
    pca = PCA(n_components=N).fit(train_data)
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    train_data_r = pca.inverse_transform(train_data_pca)
    test_data_r = pca.inverse_transform(test_data_pca)
    test_dis = []
    for i in range(len(test_data_r)):
        test_dis.append(np.sum((test_data[i]-test_data_r[i])**2)**0.5)
    score =  roc_auc_score(test_label,test_dis)
    

    draw_normal_3 = sample_by_label(train_data_r, train_label, number=10, target_label=0)
    draw_anomaly_3 = sample_by_label(test_data_r, test_label, number=10, target_label=1)
    fig_3, axs_3 = plt.subplots(2,1)
    plt.suptitle(f'{category}PCA={N}')
    axs_3[0].set_title("Anomaly Sample")
    for i in draw_anomaly_3[0]:
        axs_3[0].plot(list(range(len(draw_anomaly_3[0][0]))),i,color='red')
    axs_3[1].set_title("Normal Sample")
    for i in draw_normal_3[0]:
        axs_3[1].plot(list(range(len(draw_normal_3[0][0]))),i,color='blue')
    plt.tight_layout()
    return score

#problem 4
def discrete_fourier_transform(train_data, train_label,test_data, test_label,k = 5,M=20):

    select_index = []
    tmp = []
    for i in range(int(M/2)):
        select_index.append(i)
    if M%2 == 1:
        select_index.append(int(M/2))
    for i in range(int(M/2)):
        tmp.append(train_data.shape[1]-1-i)
    tmp.reverse()
    
    select_index.extend(tmp)

    train_data_fft = np.array([fft(row) for row in train_data])
    train_data_fft_select = train_data_fft[:,select_index].copy()
    train_data_fft_magnitude = np.abs(train_data_fft_select)

    test_data_fft = np.array([fft(row) for row in test_data])
    test_data_fft_select = test_data_fft[:,select_index].copy()
    test_data_fft_magnitude = np.abs(test_data_fft_select)

    score =  knn(train_data_fft_magnitude,test_data_fft_magnitude,test_label,k)
    train_data_ifft = np.zeros(train_data.shape, dtype=np.complex128)
    train_data_ifft[:,select_index] = train_data_fft_select
    
    test_data_ifft = np.zeros(test_data.shape, dtype=np.complex128)
    test_data_ifft[:,select_index] = test_data_fft_select


    train_data_ifft_ = np.array([np.real(ifft(row)) for row in train_data_ifft])
    test_data_ifft_ = np.array([np.real(ifft(row)) for row in test_data_ifft])
    

    draw_normal_4 = sample_by_label(train_data_ifft_, train_label, number=10, target_label=0)
    draw_anomaly_4 = sample_by_label(test_data_ifft_, test_label, number=10, target_label=1)
    fig_4, axs_4 = plt.subplots(2,1)
    plt.suptitle(f'{category}DFT={M}')
    axs_4[0].set_title("Anomaly Sample")
    for i in draw_anomaly_4[0]:
        axs_4[0].plot(list(range(len(draw_anomaly_4[0][0]))),i,color='red')
    axs_4[1].set_title("Normal Sample")
    for i in draw_normal_4[0]:
        axs_4[1].plot(list(range(len(draw_normal_4[0][0]))),i,color='blue')
    plt.tight_layout()
    return score

def haar(data,dir):
    if data.shape[1] == 2:
        left = (data[:,0]+data[:,1])/2
        right = (data[:,1]-data[:,0])/2
        left = np.reshape(left,(data.shape[0],1))
        right = np.reshape(right,(data.shape[0],1))
        return np.concatenate((left,right),axis=1)
    elif dir == "right":
        right = np.zeros((data.shape[0],data.shape[1]//2))
        for i in range(data.shape[1]//2):
            right[:,i]=(data[:,i*2+1]-data[:,i*2])/2
        return right
    else:
        left = np.zeros((data.shape[0],data.shape[1]//2))
        right = haar(data,"right")
        for i in range(data.shape[1]//2):
            left[:,i]=(data[:,i*2+1]+data[:,i*2])/2
        left_new = left.copy()
        left = haar(left_new,"left")
        return np.concatenate((left,right),axis=1)

#problem 5
def discrete_wavelet_transform(train_data, train_label,test_data, test_label,k = 5,S=32):
    level=np.ceil(np.log2(train_data.shape[1]))
    L = int(2**level)
    train_data_haar = np.zeros((train_data.shape[0],L))
    train_data_haar[:,:train_data.shape[1]] = train_data
    train_data_haar = haar(train_data_haar,"left")
    test_data_haar = np.zeros((test_data.shape[0],L))
    test_data_haar[:,:test_data.shape[1]] = test_data
    test_data_haar = haar(test_data_haar,"left")
    score =  knn(train_data_haar[:,:S],test_data_haar[:,:S],test_label,k)
    return score


if __name__=='__main__':
    # Load the data
    category = "Wafer" # Wafer / ECG200
    #category = "ECG200"
    print(f"Dataset: {category}")
    train_data = pd.read_csv(f'./{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
    test_data = pd.read_csv(f'./{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()

    train_label = train_data[:, 0].flatten()
    train_data = train_data[:, 1:]
    train_data, train_label = resample(train_data, train_label, outlier_ratio=0.0, target_label=1)

    test_label = test_data[:, 0].flatten()
    test_data = test_data[:, 1:]
    test_data, test_label = resample(test_data, test_label, outlier_ratio=0.1, target_label=1)

    
    print("-------problem 1-------")
    visualization(train_data, train_label,test_data, test_label,category)

    print("-------problem 2-------")
    k_ = [2,5,7]
    for k in k_:
        print(f"when k={k}, roc_auc_score = {knn(train_data,test_data,test_label,k)}")

    print("-------problem 3-------")
    N_ = [1,2,5,10]
    for N in N_:
        print(f"N={N}, roc_auc_score = {PCA_(train_data, train_label,test_data, test_label,k=k,N=N)}")

    print("-------problem 4-------")
    
    M_ = [10,15,20,25]
    for k in k_:
        for M in M_:
            print(f"when k={k},M={M}, roc_auc_score = {discrete_fourier_transform(train_data, train_label,test_data, test_label,k = k,M=M)}")

    print("-------problem 5-------")
    S_ = [8,16,32,64]
    for k in k_:
        for S in S_:
            print(f"when k={k},S={S}, roc_auc_score = {discrete_wavelet_transform(train_data, train_label,test_data, test_label,k = k,S=S)}")
    plt.show()
