#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:38:42 2017

@author: tdourado
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import pandas as pd



def kmeans():
    # k means determine k
    distortions = []        
    K = range(1,5)
    
    for k in K:
       kmeanModel = KMeans(n_clusters=k).fit(np.array(labsT.drop(['liquid_cost',
                          'order_id','code'], 1).astype(float)))
       kmeanModel.fit(labsT)
       distortions.append(sum(np.min(cdist(labsT, kmeanModel.cluster_centers_,
                                      'euclidean'), axis=1)) / labsT.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    
    X = np.array(labsT.drop(['liquid_cost','order_id','code'], 1).astype(float))
    y = np.array(labsT['category'])
    
    # train data 
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0,
                                                        test_size=0.50)  
    # Kmeans
    clf = KMeans(n_clusters=2, n_init=50, precompute_distances=True)
    clf.fit(X_train)
    y_pred = clf.predict(X_train)  
    centroids = clf.cluster_centers_ 
    labels = clf.labels_
    
    plt.scatter(X_train[:,0], X_train[:,1])
    plt.scatter(X_train[:,0], X_train[:,1])
    plt.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', linewidths=3,
           zorder=8)
    plt.title('kmeansCluster')
    plt.show()
    
    
# ler o  DataSet
labs = pd.read_csv("<path>/desafio.csv",
                           engine='python', encoding='latin1')
labs.convert_objects(convert_numeric=True)

# normalizaçao de transformaçao de dados 
labs.drop(['icms','tax_substitution','pis_cofins','capture_date'], 1, inplace=True)
labsT = labs.replace(['processado','captado'], ['1','0'])
labsT.process_date = labsT.process_date.replace(['0000-00-00'], ['2099-09-02'])
labsT.process_date = pd.to_datetime(labsT.process_date)
labsT.process_date = labsT.process_date.dt.month
labsT.order_status = labsT.order_status.replace(['entrega total',
                                                 'em rota de entrega',
                                                 'cancelado',
                                                 'cancelado boleto nÃ£o pago',
                                                 'solicitação de cancelamento',
                                                 'cancelado dados divergentes',
                                                 'cancelado fraude confirmada',
                                                 'cancelado não aprovado',
                                                 'disponível para retirada.',
                                                 'em rota de devolução',
                                                 'entrega parcial','entrega total',
                                                 'solicitação de troca',
                                                 'solicitaÃ§Ã£o de troca',
                                                 '-*- de troca',
                                                 '-*- de cancelamento',
                                                 'solicitação de cancelamento',
                                                 'solicitaÃ§Ã£o de cancelamento',
                                                 'fraude confirmada',
                                                 '-*- para retirada.',
                                                 'disponÃ\xadvel para retirada.',
                                                 'suspenso barragem','suspeita de fraude',
                                                 'em rota de devoluÃ§Ã£o',
                                                 'pendente processamento',
                                                 'cancelado nÃ£o aprovado'],
                                                  ['1','1','0','0','0','0','0','0',
                                                   '1','0','1','1', '0','0','0','0',
                                                   '0','0','0','1','1','0','0','0',
                                                   '0','0'] )
  
columns = ['code', 'category', 'order_id','source_channel']
    
for column in columns:
    text_digit_vals = {}
    def convert_to_int(val):
        return text_digit_vals[val]

    if labsT[column].dtype != np.int64 and labsT[column].dtype != np.float64:
            column_contents = labsT[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            labsT[column] = list(map(convert_to_int, labsT[column]))

kmeans()
