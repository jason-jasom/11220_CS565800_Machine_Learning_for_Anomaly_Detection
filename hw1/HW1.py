
########################################################
########  Do not modify the sample code segment ########
########################################################

import torchvision
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score,pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

seed = 0
np.random.seed(seed)

def resample_total(data,label,ratio=0.05):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        ratio: float, ratio of samples to be selected
    """
    new_data = []
    new_label = []
    for i in range(10):
        i_data = data[label==i]
        idx = np.random.choice(list(range(len(i_data))),int(len(i_data)*ratio),replace=False)
        new_data.append(i_data[idx])
        new_label.append(np.ones(len(idx))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

def resample(data,label,outlier_ratio=0.01,target_label=0):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        outlier_ratio: float, ratio of outliers
        target_label: int, the label to be treated as normal
    """
    new_data = []
    new_label = []
    for i in range(10):
        if i != target_label:
            i_data = data[label==i]
            target_size = len(data[label==target_label])
            num = target_size*((outlier_ratio/9))
            idx = np.random.choice(list(range(len(i_data))),int(num),replace=False)
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx))*i)
        else:
            new_data.append(data[label==i])
            new_label.append(np.ones(len(data[label==i]))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


#knn
knn_score = [[],[],[]]
k1 = [1,5,10]

#kmeans
kmeans_score = [[],[],[]]
k2 = [1,5,10]

#Distance-based
Cosine_dis_score = []
Minkowski_dis_score = [[],[],[]]
Mahalanobis_dis_score = []
r = [1,2,np.inf]

#Density-based
LOF_score = []
test_data_draw = None
test_label_draw = None
score_for_color = None

#knn
def knn(train_data,test_data,test_label,k):
    #1.
    k_near_distance_list = []
    all_data = np.concatenate((train_data,test_data))
    distance_mat = pairwise_distances(all_data,metric='euclidean')
    #2.
    for id,test in enumerate(test_data):
        distance_list = distance_mat[len(train_data)+id,:len(train_data)].copy()
        distance_list.sort()
        k_near_distance_list.append(np.mean(distance_list[:k]))
    #3.
    return roc_auc_score(test_label,k_near_distance_list)

#kmeans
def kmeans(train_data,test_data,test_label,k):
    #1.
    cluster_center_id = np.random.choice(len(train_data),k,replace=False)
    cluster_center = train_data[cluster_center_id]
    cluster_id = [0 for i in range(len(train_data))]
    #2.
    while 1:
        train_with_cluster = np.concatenate((cluster_center,train_data))
        distance_mat = pairwise_distances(train_with_cluster,metric='euclidean')
        for id,train in enumerate(train_data):
            min_dis = float('inf')
            for cluster in range(k):
                tmp = distance_mat[k+id][cluster].copy()
                if tmp < min_dis:
                    min_dis = tmp
                    cluster_id[id] = cluster
        cluster_center_tmp = np.zeros(cluster_center.shape)
        cluster_number = np.zeros(k)
        for id,cluster in enumerate(cluster_id):
            cluster_number[cluster] += 1
            cluster_center_tmp[cluster] += train_data[id]
        for cluster in range(k):
            cluster_center_tmp[cluster] /= cluster_number[cluster]
        if np.array_equal(cluster_center_tmp,cluster_center):
            break
        cluster_center = cluster_center_tmp
        cluster_center_tmp = np.zeros(cluster_center.shape)
    #3.
    k_cluster_min_distance_list = []
    test_with_cluster = np.concatenate((cluster_center,test_data))
    distance_mat = pairwise_distances(test_with_cluster,metric='euclidean')
    #4.
    for id,test in enumerate(test_data):
        min_dis = float('inf')
        for cluster in range(k):
            tmp = distance_mat[k+id][cluster].copy()
            if tmp < min_dis:
                min_dis = tmp
        k_cluster_min_distance_list.append(min_dis)
    #5.
    return roc_auc_score(test_label,k_cluster_min_distance_list)

#Distance-based
def cosine(A,B):
    cos = np.sum(A*B)/((np.sum(A**2)*np.sum(B**2))**0.5)
    return 1-cos

def minkowski(A,B,p):
    if p == np.inf:
        return np.max(abs(A-B))
    return np.sum(abs(A-B)**p)**(1/p)

def mahalanobis(A,B,S_inv):
    ans = (A-B).transpose()
    ans = ans @ S_inv
    ans = ans @ (A-B)
    return ans**0.5

def k_dis(train_data,test_data,test_label,k,distance_type=None,r=None):
    k_near_distance_list = []
    distance_mat = None
    #1.
    if distance_type == 'cosine':
        distance_mat = pairwise_distances(test_data,metric=cosine)
    elif distance_type == 'minkowski':
        distance_mat = pairwise_distances(test_data,metric=minkowski,p=r)
    elif distance_type == 'mahalanobis':
        mean= np.mean(train_data)
        S = np.zeros((train_data.shape[1],train_data.shape[1]))
        for train in train_data:
            delta = train - mean
            delta = delta.reshape((train_data.shape[1],1))
            S += delta @ delta.transpose()
        S /= train_data.shape[0]
        S_inv = np.linalg.inv(S)
        distance_mat = pairwise_distances(test_data,metric=mahalanobis,S_inv=S_inv)
    for id,test in enumerate(test_data):
        distance_list = distance_mat[id,:].copy()
        distance_list.sort()
        k_near_distance_list.append(distance_list[k])
    #2.
    return roc_auc_score(test_label,k_near_distance_list)

#Density-based

def k_distance(distance_mat,point_id,k):
    distance_list = distance_mat[point_id,:].copy()
    distance_list.sort()
    return distance_list[k]

def reachable_distance(distance_mat,point_p_id,point_o_id,k):
    return max([k_distance(distance_mat,point_o_id,k),distance_mat[point_p_id][point_o_id]])

def lrd(distance_mat,point_id,k):
    distance_list = []
    for id in range(distance_mat.shape[0]):
        distance_list.append((distance_mat[point_id][id],id))
    distance_list = sorted(distance_list, key=lambda i: i[0])
    ans = 0
    for i in range(1,k+1):
        ans += reachable_distance(distance_mat,point_id,distance_list[i][1],k)
    return 1/(ans/k)

def lof(distance_mat,point_id,k):
    distance_list = []
    for id in range(distance_mat.shape[0]):
        distance_list.append((distance_mat[point_id][id],id))
    distance_list = sorted(distance_list, key=lambda i: i[0])
    ans = 0
    for i in range(1,k+1):
        ans += lrd(distance_mat,distance_list[i][1],k)
    ans /= lrd(distance_mat,point_id,k)
    return ans/k

def LOF(test_data,test_label,k,drawing=False):
    #1.
    distance_list = []
    score_for_color_tmp = []
    distance_mat = pairwise_distances(test_data,metric='euclidean')
    for id,test in enumerate(test_data):
        score = lof(distance_mat,id,k)
        distance_list.append(score)
        score_for_color_tmp.append(score)
    #2.
    if drawing:
        global test_data_draw
        global test_label_draw
        global score_for_color
        test_data_draw = TSNE(n_components=2).fit_transform(test_data)
        test_label_draw = test_label.copy()
        score_for_color = score_for_color_tmp.copy()
    #3.
    return roc_auc_score(test_label,distance_list)

if __name__=="__main__":
    orig_train_data = torchvision.datasets.MNIST("MNIST/", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),target_transform=None,download=True) #下載並匯入MNIST訓練資料
    orig_test_data = torchvision.datasets.MNIST("MNIST/", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),target_transform=None,download=True) #下載並匯入MNIST測試資料

    orig_train_label = orig_train_data.targets.numpy()
    orig_train_data = orig_train_data.data.numpy()
    orig_train_data = orig_train_data.reshape(60000,28*28)

    orig_test_label = orig_test_data.targets.numpy()
    orig_test_data = orig_test_data.data.numpy()
    orig_test_data = orig_test_data.reshape(10000,28*28)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30)
    pca_data = pca.fit_transform(np.concatenate([orig_train_data,orig_test_data]))
    orig_train_data = pca_data[:len(orig_train_label)]
    orig_test_data = pca_data[len(orig_train_label):]

    orig_train_data,orig_train_label = resample_total(orig_train_data,orig_train_label,ratio=0.1)

    for i in tqdm.tqdm(range(10)):
        train_data = orig_train_data[orig_train_label==i]
        test_data,test_label = resample(orig_test_data,orig_test_label,target_label=i,outlier_ratio=0.1)
        # [TODO] prepare training/testing data with label==i labeled as 0, and others labeled as 1
        test_label_01 = np.zeros(test_data.shape[0])
        for j in range(len(test_label_01)):
            if test_label[j] != i:
                test_label_01[j] = 1
        test_label = test_label_01
        # [TODO] implement methods
        # [TODO] record ROC-AUC for each method
        for j in range(len(k1)):
            knn_score[j].append(knn(train_data,test_data,test_label,k=k1[j]))
        for j in range(len(k2)):
            kmeans_score[j].append(kmeans(train_data,test_data,test_label,k=k2[j]))
        Cosine_dis_score.append(k_dis(train_data,test_data,test_label,k=5,distance_type='cosine',r=None))
        for j in range(len(r)):
            Minkowski_dis_score[j].append(k_dis(train_data,test_data,test_label,k=5,distance_type='minkowski',r=r[j]))
        Mahalanobis_dis_score.append(k_dis(train_data,test_data,test_label,k=5,distance_type='mahalanobis',r=None))
        LOF_score.append(LOF(test_data,test_label,k=5,drawing=(i==0)))
        
# [TODO] print the average ROC-AUC for each method
print("------Problem 1------")
for i in range(len(k1)):
    print(f"knn with k = {k1[i]}, score = {np.mean(knn_score[i])}")

print("------Problem 2------")
for i in range(len(k2)):
    print(f"kmeans with k = {k2[i]}, score = {np.mean(kmeans_score[i])}")

print("------Problem 3------")
print(f"Cosine_distance, score = {np.mean(Cosine_dis_score)}")
for i in range(len(r)):
    print(f"Minkowski_distance with r = {r[i]}, score = {np.mean(Minkowski_dis_score[i])}")
print(f"Mahalanobis_distance, score = {np.mean(Mahalanobis_dis_score)}")

print("------Problem 4------")
print(f"Local Outlier Factor, score = {np.mean(LOF_score)}")

fig, axs = plt.subplots(1,2)

axs[0].set_title("predicted LOF score for normal digit=0")
x_val = [test_data_draw[i][0] for i in range(len(test_label_draw))]
y_val = [test_data_draw[i][1] for i in range(len(test_label_draw))]
sc = axs[0].scatter(x_val,y_val,c=score_for_color)
axs[1].set_title("ground truth label for normal digit=0")
indices0 = [i for i in range(len(test_label_draw)) if test_label_draw[i] == 0]
indices1 = [i for i in range(len(test_label_draw)) if test_label_draw[i] == 1]
normal_x = [test_data_draw[i][0] for i in indices0]
normal_y = [test_data_draw[i][1] for i in indices0]
anomaly_x = [test_data_draw[i][0] for i in indices1]
anomaly_y = [test_data_draw[i][1] for i in indices1]
axs[1].scatter(normal_x,normal_y,color="blue",label='normal')
axs[1].scatter(anomaly_x,anomaly_y,color="orange",label='anomaly')
axs[1].legend()

plt.colorbar(sc)
plt.tight_layout()
plt.show()