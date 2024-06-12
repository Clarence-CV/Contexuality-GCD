# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# import torch_clustering
#
# class Keans_Loss(nn.Module):
#     def __init__(self, temperature = 0.5, n_cluster = 196):
#         super(Keans_Loss,self).__init__()
#         self.temperature = temperature
#         self.n_cluster = n_cluster
#         self.clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4,
#                                                                n_clusters=self.num_cluster)
#         self.psedo_labels = None
#
#         def clustering(self, n_cluster, features):
#
#             clustering_model =  torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4,
#                                                                n_clusters=self.num_cluster)
#             psedo_labels =  clustering_model.fit_predict(features)
#             self.psedo_labels = psedo_labels
#             cluster_centers = clustering_model.cluster_centers_
#             return psedo_labels, cluster_centers
#
#         def compute_cluster_loss(self,
#
#
#
import torch

x = torch.rand((6, 10))  # 假设有一个形状为 (6, 10) 的张量
print(x)
print("==============")
proj,out = x.chunk(2)
print(proj)

print("============")

print(out)
print("==============")
print(out[0].argmax(1))