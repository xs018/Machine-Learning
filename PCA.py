import pandas as pd
import numpy as np
import os
import random
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

np.random.seed(0)

# data_path_base = "/ocean/projects/mch210006p/shared/HW1/Classification"
# post_path = os.path.join(data_path_base, "post-CHF")
# pre_path = os.path.join(data_path_base, "pre-CHF")
# post_dirs = glob(f"{post_path}/*.jpg")
# pre_dirs = glob(f"{pre_path}/*.jpg")

# dataset = []

# for dir in post_dirs:
#     dataset.append([dir, 1])

# for dir in pre_dirs:
#     dataset.append([dir, 0])

# dataset = np.asarray(dataset)
# print(dataset.shape)
# df = pd.DataFrame(dataset)
# df.to_csv("dataset.csv", index=False, header=["path", "label"])

n_samples = 23890
# print(len(dataset))

image_size = (240, 240)
dataset = pd.read_csv("dataset.csv")
idx = np.random.choice(len(dataset), n_samples)
selected = dataset.iloc[idx]
data = np.empty((n_samples, image_size[0]*image_size[1]))
for i in range(n_samples):
    path = selected.iloc[i, 0]
    img = np.float32(imread(path)) / 255.
    image_resized = resize(img, image_size, anti_aliasing=True)
    data[i] = image_resized.flatten()
print(data.shape)

split_ratio = 0.2
test_num = int(n_samples*split_ratio)
train_num = n_samples - test_num
x_train = data[:train_num, :]
x_test =  data[train_num:, :]


def fit_pca(n_components):
    # sc = StandardScaler()
    # sc.fit(data)
    # data_std = sc.transform(data)

    # Instantiate PCA
    pca = PCA(n_components=n_components)

    # Determine transformed features
    train_pcs = pca.fit_transform(x_train)

    ###################################### prob 1 ######################################
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var_pca = pca.explained_variance_ratio_
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # Create the visualization plot
    plt.figure()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"res/hw3/1_{n_components}_train{train_num}_val{test_num}.jpg")

    ###################################### prob 2 ######################################
    sample = x_test[1].reshape(1, -1)
    sample_pcs = pca.transform(sample)
    projected_sample = pca.inverse_transform(sample_pcs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sample.reshape(image_size), cmap="gray")
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(projected_sample.reshape(image_size), cmap="gray")
    plt.title("Reconstructed")
    plt.axis('off')
    plt.savefig(f"res/hw3/2_{n_components}_train{train_num}_val{test_num}.jpg")

    ###################################### prob 3 ######################################
    test_pcs = pca.transform(x_test)
    projected_test = pca.inverse_transform(test_pcs)
    err = ((projected_test - x_test) ** 2).mean()

    reconstruction_err.append(err)

    ###################################### prob 4 ######################################
    n_clusters = 2
    k_means = KMeans(n_clusters=n_clusters, random_state=0)
    k_means.fit(train_pcs)
    centroids = k_means.cluster_centers_
    label = k_means.labels_

    if n_components > 1:
        plt.figure()
        for i in range(n_clusters):
            plt.scatter(train_pcs[label == i, 0], train_pcs[label == i, 1], edgecolor='none', c=np.random.rand(1, 3),
                        label=f"Cluster {i + 1}")

        plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=100, c="r", label="Cluster Centroid")
        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f"K-Means Clustering Results with K={n_clusters}")
        plt.savefig(f"res/hw3/4_{n_components}_train{train_num}_val{test_num}.jpg")

n_pcomponents = [1, 2, 10, 20, 50, 100]

reconstruction_err = []
for components in n_pcomponents:
    fit_pca(components)

plt.figure()
plt.plot(n_pcomponents, reconstruction_err)
plt.savefig(f"res/hw3/3_train{train_num}_val{test_num}.jpg")


