import numpy as np
import pandas as pd
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import scipy
from scipy import io
import sklearn
from sklearn import preprocessing
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rotateImage import Rotate

class cifarLoader(object):
    def __init__(self, labeldSamples, unlabeldSamples, testingBatchSize = 1000, nEachClassSamples = None):
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)
        self.batchSize  = testingBatchSize
        self.loadData(labeldSamples, unlabeldSamples, nEachClassSamples)
        
    def loadData(self, labeldSamples, unlabeldSamples, nEachClassSamples):
        
#         pdb.set_trace()
        # labeled data
        labeledData  = torch.Tensor(labeldSamples.data.shape[0],1,32,32)
        labeledLabel = torch.LongTensor(labeldSamples.data.shape[0])

        for idx, example in enumerate(labeldSamples):
#             pdb.set_trace()
            labeledData[idx]  = example[0][2:3] #torch.unsqueeze(torch.mean(example[0],0), 0) # example[0][2:3]
            labeledLabel[idx] = example[1]
            
        self.labeledData  = labeledData
        self.labeledLabel = labeledLabel

        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        self.nLabeledData = self.labeledData.shape[0]
        
        if nEachClassSamples is not None:
            selectedLabeledData  = torch.Tensor(self.nOutput*nEachClassSamples,1,32,32)
            selectedLabeledLabel = torch.LongTensor(self.nOutput*nEachClassSamples)

            idx = 0
            selectedDataIdx = []
            for iClass in self.classes:
                # print(iClass)
                dataCount = 0

                for iData in range(0,self.nLabeledData):
                    # print(iData)
                    if labeledLabel[iData] == iClass:
                        selectedLabeledData[idx]  = self.labeledData[iData]
                        selectedLabeledLabel[idx] = self.labeledLabel[iData]
                        # print(labeledLabel[iData])
                        idx += 1
                        dataCount += 1

                        selectedDataIdx.append(iData)
                        iData += 1

                    if dataCount == nEachClassSamples:
                        break

            remainderData     = deleteRowTensor(self.labeledData, selectedDataIdx, 2)
            remainderLabel    = deleteRowTensor(self.labeledLabel, selectedDataIdx, 2)
            
            self.nLabeledData = selectedLabeledData.shape[0]

            # shuffle
            indices = torch.randperm(self.nLabeledData)
            self.labeledData  = selectedLabeledData[indices]
            self.labeledLabel = selectedLabeledLabel[indices]
        
        # unlabeled data
        unlabeledData  = torch.Tensor(unlabeldSamples.data.shape[0],1,32,32)
        unlabeledLabel = torch.LongTensor(unlabeldSamples.data.shape[0])

        for idx, example in enumerate(unlabeldSamples):
            unlabeledData[idx]  = example[0][2:3] #torch.unsqueeze(torch.mean(example[0],0), 0) # example[0][2:3]
            unlabeledLabel[idx] = example[1]
        
        if nEachClassSamples is not None:
            unlabeledData   = torch.cat((unlabeledData,remainderData),0)
            unlabeledLabel  = torch.cat((unlabeledLabel,remainderLabel),0)

        self.unlabeledData  = unlabeledData
        self.unlabeledLabel = unlabeledLabel
        
        self.nUnlabeledData = self.unlabeledData.shape[0]
        self.nBatch         = int(self.nUnlabeledData/self.batchSize)
        self.taskIndicator  = (torch.zeros(self.nBatch).long()).tolist()
        
        print('Number of output: ', self.nOutput)
        print('Number of labeled data: ', self.nLabeledData)
        print('Number of unlabeled data: ', self.nUnlabeledData)
        print('Number of unlabeled data batch: ', self.nBatch)

    def createTask(self, nTask = 2, taskList = [], taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2 or taskType == 3) and (len(taskList) != nTask):
            raise NameError('list of rotaion angle should be the same with the number of task')

        self.taskIndicator        = []

        # clone labeled data
        transformedLabeledData    = self.labeledData.clone()
        transformedLabeledLabel   = self.labeledLabel.clone()
        finalLabeledData          = {}
        finalLabeledLabel         = {}

        # clone unlabeled data
        transformedUnlabeledData  = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData        = {}
        finalUnlabeledLabel       = {}

        # testing data
        unlabeledDataTest        = {}
        unlabeledLabelTest       = {}
        
        # number of data and batch for each task
        self.nTask                  = nTask
        self.nLabeledDataPerTask    = int(self.nLabeledData/nTask)
        self.nBatchPerTask          = int(self.nBatch/nTask)
        self.nBatch                 = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize

        nUnlabeledDataTest = 0

        for iTask in range(0,nTask):
            # load data
            # iTask = iTask + 1
            

            # load labeled data
            taskLabeledData    = transformedLabeledData[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]
            taskLabeledLabel   = transformedLabeledLabel[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]

            # load unlabeled data
            taskUnlabeledData  = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0] or self.nLabeledDataPerTask != taskLabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskLabeledData   = taskLabeledData.view(taskLabeledData.size(0),-1)
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute labeled data
                taskLabeledData   = taskLabeledData[:, torch.tensor(col_idxs)]

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskLabeledData   = taskLabeledData.reshape(taskLabeledData.size(0),1,32,32)
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),1,32,32)

            elif taskType == 2:
                # rotate labeled data
                for idx, _ in enumerate(taskLabeledData):
                    taskLabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskLabeledData[idx])

                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskUnlabeledData[idx])                    
            
            elif taskType == 3:
                # split MNIST, into 5 tasks
                self.nOutputPerTask = 2

                taskLabeledDataSplit    = torch.Tensor().float()
                taskLabeledLabelSplit   = torch.Tensor().long()

                taskUnlabeledDataSplit  = torch.Tensor().float()
                taskUnlabeledLabelSplit = torch.Tensor().long()

                for iClass in taskList[iTask]:
                    # split labeled data
                    taskLabeledDataSplit  = torch.cat((taskLabeledDataSplit,transformedLabeledData[transformedLabeledLabel==iClass]),0)
                    taskLabeledLabelSplit = torch.cat((taskLabeledLabelSplit,transformedLabeledLabel[transformedLabeledLabel==iClass]),0)

                    # split unlabeled data
                    taskUnlabeledDataSplit  = torch.cat((taskUnlabeledDataSplit,transformedUnlabeledData[transformedUnlabeledLabel==iClass]),0)
                    taskUnlabeledLabelSplit = torch.cat((taskUnlabeledLabelSplit,transformedUnlabeledLabel[transformedUnlabeledLabel==iClass]),0)

                # shuffle labeled data
                taskLabeledData  = taskLabeledDataSplit
                taskLabeledLabel = taskLabeledLabelSplit
                
                row_idxs = list(range(taskLabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskLabeledData  = taskLabeledData[torch.tensor(row_idxs), :]
                taskLabeledLabel = taskLabeledLabel[torch.tensor(row_idxs)]

                # shuffle unlabeled data
                taskUnlabeledData  = taskUnlabeledDataSplit
                taskUnlabeledLabel = taskUnlabeledLabelSplit

                row_idxs = list(range(taskUnlabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskUnlabeledData  = taskUnlabeledData[torch.tensor(row_idxs), :]
                taskUnlabeledLabel = taskUnlabeledLabel[torch.tensor(row_idxs)]

            # store labeled data and labels
            finalLabeledData[iTask]  = taskLabeledData
            finalLabeledLabel[iTask] = taskLabeledLabel

            # store unlabeled data and labels
            finalUnlabeledData[iTask]  = taskUnlabeledData[self.batchSize:]
            finalUnlabeledLabel[iTask] = taskUnlabeledLabel[self.batchSize:]
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(int(finalUnlabeledData[iTask].shape[0]/self.batchSize)).long()).tolist()

            # store unlabeled data for testing
            unlabeledDataTest[iTask]  = taskUnlabeledData[:self.batchSize]
            unlabeledLabelTest[iTask] = taskUnlabeledLabel[:self.batchSize]  
            nUnlabeledDataTest += unlabeledDataTest[iTask].shape[0]        


        # labeled data
        self.labeledData    = finalLabeledData
        self.labeledLabel   = finalLabeledLabel

        # unlabeled data
        self.unlabeledData  = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest  = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest

        # update size
        self.nBatchPerTask          = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)
        self.nBatch                 = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask  = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)*self.batchSize
        self.nUnlabeledDataTest     = nUnlabeledDataTest

        print('Number of task: ', nTask)
        print('Number of labeled data per task: ', self.nLabeledDataPerTask)
        print('Number of unlabeled data per task: ', self.nUnlabeledDataPerTask)
        print('Number of unlabeled data batch per task: ', self.nBatchPerTask)
        print('Number of unlabeled data test: ', self.nUnlabeledDataTest)

    def createDrift(self, nDrift = 2, taskList = [], taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2) and (len(taskList) != nDrift):
            raise NameError('list of rotaion angle should be the same with the number of task')

        # clone unlabeled data
        transformedUnlabeledData  = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData        = torch.Tensor().float()
        finalUnlabeledLabel       = torch.Tensor().long()
        
        # number of data and batch for each task
        self.nDrift                  = nDrift
        self.nBatchPerTask          = int(self.nBatch/nDrift)
        self.nBatch                 = self.nBatchPerTask*nDrift
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize

        for iTask in range(0,nDrift):
            # load data
            # iTask = iTask + 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(self.nBatchPerTask).long()).tolist()

            # load unlabeled data
            taskUnlabeledData  = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),1,32,32)

            elif taskType == 2:
                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskUnlabeledData[idx])

            # store unlabeled data and labels
            finalUnlabeledData  = torch.cat((finalUnlabeledData,taskUnlabeledData),0)
            finalUnlabeledLabel = torch.cat((finalUnlabeledLabel,taskUnlabeledLabel),0)

        # unlabeled data
        self.unlabeledData  = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel