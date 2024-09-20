import numpy as np
import os
import pickle
import constants
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import random
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from collections import defaultdict


def random_subset(input_array, n):
    """
    Selects a random subset of length n from the input_array.
    
    Args:
        input_array (list): The input array from which to select the subset.
        n (int): The desired length of the random subset.

    Returns:
        list: A random subset of input_array with length n.
    """
    if n < 0 or n > len(input_array):
        raise ValueError("Invalid value of n")

    return random.sample(input_array, n)

def roundto5s(input):
    output = np.array(np.round(input / 5, 0) * 5)
    return output

def mean_abs_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def save_np(array, path, saveanyway=True):
    if (not os.path.isfile(path) or saveanyway):
        f = open(path, 'wb')
        np.save(f,array)
        #pickle.dump(X.columns,f)
        f.close()

def save_pickle(array, path, saveanyway=True):
    if (not os.path.isfile(path) or saveanyway):
        f = open(path, 'wb')
        pickle.dump(array,f)
        f.close()

def load_pickle(path):
    file = open(path,'rb')
    array = pickle.load(file)
    file.close()
    return array

def predict_and_round(model, x_data):
    predicted = model.predict(x_data)
    predicted_rounded = roundto5s(predicted)
    return predicted_rounded

def distance_filter(all_columns, selected_cols, coefficients, close_features):
    
    for feature_pair in close_features:
        if feature_pair[0] not in selected_cols or feature_pair[1] not in selected_cols:
            continue
        coef1 = coefficients[all_columns.get_loc(feature_pair[0])]
        print(feature_pair[0]) 
        print(coef1)
        coef2 = coefficients[all_columns.get_loc(feature_pair[1])]
        print(feature_pair[1])
        print(coef2)
        if np.abs(coef1) > np.abs(coef2):
            selected_cols = selected_cols.drop(feature_pair[1])
        else:
            selected_cols = selected_cols.drop(feature_pair[0])
    return selected_cols

def correlation_cluster(x_train, cluster_n_bool=False, n_clusters=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #x_train = x_train.drop(columns=["Gender-0=M, 1 =F","Side 0=L, 1=R"])
    #print(x_train)
    corr = spearmanr(x_train).correlation
    
    # remove age and gender
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    
    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    print(dist_linkage.shape)
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=x_train.columns.to_list(), ax=ax1, leaf_rotation=90
    )
    #print(dendro["ivl"].index("Age"))
    # a = dendro["ivl"].index("3000BefBone")
    # b = dendro["ivl"].index("4000BefBone")
    # corr1=distance_matrix[dendro["leaves"],:][:, dendro["leaves"]]
    # print(corr1[a][b])
    # count = np.sum(distance_matrix < 0.12)
    # print(count)
    # count1 = np.sum(distance_matrix < 1.1)
    # print(count1)
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    
    img = ax2.imshow(corr[dendro["leaves"],:][:, dendro["leaves"]])
    plt.colorbar(img, ax=ax2)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    _ = fig.tight_layout()

    if not cluster_n_bool:
        cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        
    else:
        cluster_ids = hierarchy.fcluster(dist_linkage, n_clusters, criterion="maxclust")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
    return cluster_id_to_feature_ids


def correlation_distance(x_train):
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #x_train = x_train.drop(columns=["Gender-0=M, 1 =F","Side 0=L, 1=R"])
    #print(x_train)
    corr = spearmanr(x_train).correlation
    
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    
    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    print(dist_linkage.shape)
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=x_train.columns.to_list(), no_plot=True, leaf_rotation=90
    )
    distances=distance_matrix[dendro["leaves"],:][:, dendro["leaves"]]
    indices = np.where(distances < 0.12)
    
    pairs = list(zip(indices[0], indices[1]))
    i=0
    while i != len(pairs):
        if not pairs[i][0] < pairs[i][1]:
            pairs.pop(i)
        else:
            i += 1

    col_pairs =[]
    for pair in pairs:
        col_pair = [0,0]
        col_pair[0] = dendro["ivl"][pair[0]]
        col_pair[1] = dendro["ivl"][pair[1]]
        col_pairs.append(col_pair)

    return col_pairs


    

def correlation_select(x_train):
    cluster_id_to_feature_ids = correlation_cluster(x_train)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = x_train.columns[selected_features]

    return selected_features_names

def main():

    return 

if __name__ == "__main__":
    main()

