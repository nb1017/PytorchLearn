from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import pandas as pd
import re as re

class Titanic(torch.utils.data.Dataset):
    def __init__(self, transforms=None,train=True):
        self.transforms=transforms
        self.t=train
        self.train=pd.read_csv('./input/train.csv', header=0,dtype={'Age':np.float64})
        self.test=pd.read_csv('./input/test.csv', header=0, dtype={'Age':np.float64})
        self.full_data=[self.train,self.test]

        for dataset in self.full_data:
            dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
        for dataset in self.full_data:
            dataset['IsAlone']=0
            dataset.loc[dataset['FamilySize']==1,'IsAlone']=1

        for dataset in self.full_data:
            dataset['Embarked']=dataset['Embarked'].fillna('S')
        for dataset in self.full_data:
            dataset['Fare']=dataset['Fare'].fillna(self.train['Fare'].median())
        self.train['CategoricalFare']=pd.qcut(self.train['Fare'],4)
        for dataset in self.full_data:
            age_avg=dataset['Age'].mean()
            age_std=dataset['Age'].std()
            age_null_count=dataset['Age'].isnull().sum()
            age_null_random_list=np.random.randint(age_avg-age_std,
                                                   age_avg+age_std,
                                                   size=age_null_count)
            dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list
            dataset['Age']=dataset['Age'].astype(int)
        self.train['CategoricalAge']=pd.cut(self.train['Age'],5)

        def get_title(name):
            title_search=re.search(' ([A-Za-z]+)\.',name)
            if title_search:
                return title_search.group(1)
            return ""
        for dataset in self.full_data:
            dataset['Title']=dataset['Name'].apply(get_title)

        for dataset in self.full_data:
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        for dataset in self.full_data:
            dataset['Sex']=dataset['Sex'].map({'female':0, 'male':1}).astype(int)
            title_mapping={"Mr":1, 'Miss':2, 'Mrs':3, "Master":4, 'Rare':5}
            dataset['Title']=dataset['Title'].map(title_mapping)
            dataset['Title']=dataset['Title'].fillna(0)
            dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

            dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
            dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
            dataset['Fare'] = dataset['Fare'].astype(int)

            # Mapping Age
            dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[dataset['Age'] > 64, 'Age'] = 4

        drop_elements=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
        self.train=self.train.drop(drop_elements,axis=1)
        self.train=self.train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

        self.test=self.test.drop(drop_elements,axis=1)

        self.train=self.train.values
        self.test=self.test.values

    def __getitem__(self, item):
        if self.t:
            return self.train[item][1:],self.train[item][0]
        else :
            return self.test[item][1:],self.test[item][0]

    def __len__(self):
        return len(self.train)

class classifier(nn.Module):
    def __init__(self):
        super(classifier,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(7,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.layer(x)
        return x

device='cuda' if torch.cuda.is_available() else 'cpu'
model=classifier().to(device)
# model=nn.Sequential(
#     nn.Linear(7,128),
#     nn.ReLU(),
#     nn.Linear(128,1),
#     nn.Sigmoid()
#     ).to(device)
criterion=nn.BCELoss()
# optimizer=optim.SGD(model.parameters(),momentum=0.9, lr=1e-3)
optimizer=optim.Adam(model.parameters())
# def train(epoch):
#     model.train()
#     for i , data,target in enumerate(train_loader):
#         data,target=data.to(device), target.to(device)
#         output=model(data)
#
#         optimizer.zero_grad()
#         loss=criterion(output, target)
#         loss.backward()
#         optimizer.step()
#
#         if i%50==0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,i*len(data,len(train_loader.dataset),100.)))


if __name__=="__main__":
    train=Titanic()
    test=Titanic(train=False)
    dieta_loader=torch.utils.data.DataLoader(dataset=train, batch_size=16, shuffle=True)
    dieta_loader_test=torch.utils.data.DataLoader(dataset=test, batch_size=16, shuffle=False)
    for epoch in range(200):
        for features, labels in dieta_loader:
            features,labels=features.type(torch.FloatTensor).to(device),labels.type(torch.FloatTensor).to(device)
            outputs=model(features)
            loss=criterion(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct=0
        total=0
        for features, labels in dieta_loader:
            features=features.type(torch.FloatTensor).to(device)
            outputs = model(features)
            predicted=outputs
            for i in range(len(predicted)):
                if predicted[i]>=0.5:
                    predicted[i]=1
                else :
                    predicted[i]=0
            predicted.view(1,-1)
            total+=labels.size(0)
            correct+=predicted.type(torch.LongTensor).eq(labels.data.view_as(predicted)).sum()
        print('Accuracy : {}'.format(100*correct/total))
