#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:36:41 2017

@author: tdourado
"""


import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

def Cnn():
    
    X = np.array(labsT.drop(['order_id','quantity'], 1).astype(float))
    y = np.array(labsT['quantity'])
    
    # train data 
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                         test_size=0.2) 
    # Kmeans
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Acuracia do cluster :",accuracy)

    predictme = clf.predict(X)      
#    print(X)
#    print(predictme)
#    print(X_train)
    
#    df = pd.DataFrame(predictme)
#    df.to_csv("<path>/predict_qtF.csv")
#    df2 = pd.DataFrame(X_train)
#    df2.to_csv("<path>/sample.csv")
#    df3 = pd.DataFrame(X)
#    df3.to_csv("<path>/X.csv")
    
    
# ler o  DataSet
luiza_labs = pd.read_csv("<path>/desafio.csv",
                           engine='python', encoding='latin1')
luiza_labs.convert_objects(convert_numeric=True)

# normalizaçao de transformaçao de dados 
luiza_labs.drop(['icms','tax_substitution','pis_cofins','capture_date','liquid_cost'], 1, inplace=True)
labsT = luiza_labs.replace(['processado','captado'], ['1','0'])
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
  
columns = ['source_channel', 'category', 'order_id','code']
    
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
#            df4 = pd.DataFrame(labsT[column])
#            df4.to_csv("<path>/transf_dictionaryCode.csv")
            labsT[column] = list(map(convert_to_int, labsT[column]))
#            df5 = pd.DataFrame(labsT[column])
#            df5.to_csv("<path>/transf_dictionary.csv")
           
Cnn()