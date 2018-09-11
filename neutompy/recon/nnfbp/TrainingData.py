#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pynnfbp/
#
#
#This file is part of the PyNN-FBP, a Python implementation of the
#NN-FBP tomographic reconstruction method.
#
#PyNN-FBP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PyNN-FBP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyNN-FBP. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import tempfile
import numpy as np

import scipy.io as sio

import sys
import os
import math
import random
class TrainingData(object):
    '''Base object of a class that represents training or validation data used
    during training of a network.
    
    An implementing class should define ``getDataBlock``, ``addDataBlock`` and ``normalizeData`` methods. See,
    for example, :class:`HDF5TrainingData`.
    
    :param data: Dataset to pick pixels from. (see :mod:`nnfbp.DataSet`)
    :type data: DataSet
    :param nPoints: Number of pixels to pick.
    :type nPoints: :class:`int`
    :param blockSize: Size of each data block.
    :type blockSize: :class:`int`
    '''
    
    def __setupIDX(self,size):
        '''Create a variable ``idx`` that gives location of pixels that can be picked.'''
        ym,xm = np.ogrid[-(size-1.)/2.:(size-1.)/2.:complex(0,size),-(size-1.)/2.:(size-1.)/2.:complex(0,size)]
        bnd = (size)**2/4
        self.mask = xm**2+ym**2<=bnd
        x,y = np.where(self.mask==True)
        self.idx = zip(x,y)
    
    def __getPickedIndices(self,nToPick):
        '''Return a list of the location of ``nToPick`` randomly selected pixels.'''
        nTimesToDo = int(math.ceil(nToPick/float(len(self.idx))))
        iList = []
        for i in range(nTimesToDo):
            iList.extend(self.idx)
        return zip(*random.sample(iList,nToPick))
    
    def __getPickedIndicesWithMask(self,nToPick,mask):
        '''Return a list of the location of ``nToPick`` randomly selected pixels.'''
        maskCombined = self.mask+mask
        x,y = np.where(maskCombined>1)
        idx = zip(x,y)
        nTimesToDo = int(math.ceil(nToPick/float(len(idx))))
        iList = []
        for i in range(nTimesToDo):
            iList.extend(idx)
        return zip(*random.sample(iList,nToPick))
    
    def __init__(self,data,nPoints,network,blockSize=10000):
        tmpFl = tempfile.mkstemp(dir=network.tmpDir)
        self.fn= tmpFl[1]
        os.close(tmpFl[0])
        pickArray = np.histogram(np.floor(data.nImages*np.random.rand(nPoints)), data.nImages, (0,data.nImages))[0]
        if np.max(pickArray)>blockSize:
            raise Exception('Buffer size is too small!')
        nParameters = network.red.outSize
        self.nPar = nParameters
        curData = np.empty((np.max(pickArray),nParameters+1))
        outData = np.empty((blockSize,nParameters+1))
        self.__setupIDX(network.proj.recSize)
        self.nBlocks=0
        nInBlock=0
        i=0
        for i in range(len(data)):
            example = data[i]
            nToPick = pickArray[i]
            
            if nToPick==0: continue
            image = example[0]
            sino = example[1]
            angles = example[2]
            if len(example)>3:
                pickedIndices = self.__getPickedIndicesWithMask(nToPick,example[3])
            else:
                pickedIndices = self.__getPickedIndices(nToPick)
            for j in range(nParameters):
                backImage = network.proj.reconstructWithFilter(sino,network.red.filters[:,j])
                curData[:nToPick,j] = backImage[pickedIndices]
            curData[:nToPick,nParameters] = image[pickedIndices]
            if nInBlock+nToPick<blockSize:
                outData[nInBlock:nInBlock+nToPick,:]=curData[:nToPick,:].copy()
                nInBlock+=nToPick
            else:
                nToWrite = blockSize-nInBlock
                nLeft = nToPick - nToWrite
                if nToWrite>0:
                    outData[nInBlock:blockSize,:] = curData[0:nToWrite,:].copy()
                    self.addDataBlock(outData,self.nBlocks)
                    self.nBlocks+=1
                nInBlock=0
                if nLeft>0:
                    outData[0:nLeft,:] = curData[nToWrite:nToPick,:].copy()
                    nInBlock+=nLeft
            percDone = float(blockSize*self.nBlocks + nInBlock)/nPoints
            nTicksDone = (int)(percDone*60)
            sys.stdout.write('\r[%s>%s] %d%% %s' % ('-'*nTicksDone, ' '*(60-nTicksDone), 100*percDone,50*' '))             
            sys.stdout.flush()
        if nInBlock>0:
            self.addDataBlock(outData[0:nInBlock,:], self.nBlocks)
            self.nBlocks+=1
        sys.stdout.write('\n')
        sys.stdout.flush()
    
    def addDataBlock(self,data,i):
        '''Add a block of data to the set.
        
        :param data: Block of data to add.
        :type data: :class:`numpy.ndarray`
        :param i: Position to add block to.
        :type i: :class:`int`
        '''
        raise NotImplementedError("TrainingData: Subclass should implement this method.")
    
    def getDataBlock(self,i):
        '''Get a block of data from the set.
        
        :param i: Position of block to get.
        :type i: :class:`int`
        :returns: :class:`numpy.ndarray` -- Block of data.
        '''
        raise NotImplementedError("TrainingData: Subclass should implement this method.")
    
    def getMinMax(self):
        '''Returns the minimum and maximum values of each column of the entire set.
        
        :returns: :class:`tuple` with:
        - ``minL`` -- :class:`numpy.ndarray` with minimum value of each column except last.
        - ``maxL`` -- :class:`numpy.ndarray` with maximum value of each column except last.
        - ``minIn`` -- :class:`float` minimum values of last column.
        - ``maxIn`` -- :class:`float` maximum values of last column.
        '''
        minL = np.empty(self.nPar)
        minL.fill(np.inf)
        maxL = np.empty(self.nPar)
        maxL.fill(-np.inf)
        maxIn = -np.inf
        minIn = np.inf
        for i in range(self.nBlocks):
            data = self.getDataBlock(i)
            if data is None:
                continue
            maxL = np.maximum(maxL, data[:, 0:self.nPar].max(0))
            minL = np.minimum(maxL, data[:, 0:self.nPar].min(0))
            maxIn = np.max([maxIn, data[:, self.nPar].max()])
            minIn = np.min([minIn, data[:, self.nPar].min()])
        return (minL,maxL,minIn,maxIn)
    
    def normalizeData(self,minL,maxL,minIn,maxIn):
        '''Normalize the set such that every column is in range (0,1), except for the last column,
        which will be normalized to (0.25,0.75). Parameters are like ``getMinMax()``.
        '''
        raise NotImplementedError("TrainingData: Subclass should implement this method.")
    
    def close(self):
        '''Close the underlying file.'''
        os.remove(self.fn)

class MATTrainingData(TrainingData):
    '''Implementation of :class:`TrainingData` that uses a MAT files to store data.
    '''
    
    def getDataBlock(self,i):
        if not self.tileM is None:
            data = sio.loadmat(self.fls[i])[self.dataname]
            data[:,0:self.nPar] =2*(data[:,0:self.nPar]-self.tileM)/self.maxmin - 1
            data[:,self.nPar] = 0.25+(data[:,self.nPar]-self.minIn)/(2*(self.maxIn-self.minIn))
            return data
        else:
            return sio.loadmat(self.fls[i])[self.dataname]
    
    def normalizeData(self,minL,maxL,minIn,maxIn):
        data = sio.loadmat(self.fls[0])[self.dataname]
        self.tileM = np.tile(minL, (data.shape[0],1))
        self.maxmin = np.tile(maxL-minL, (data.shape[0],1))
        self.minIn = minIn
        self.maxIn = maxIn
    
    def __init__(self,fls,dataname='mat'):
        self.fls = fls
        self.nBlocks = len(fls)
        self.dataname = dataname
        self.tileM = None
        self.nPar = (sio.loadmat(self.fls[0])[self.dataname]).shape[1]-1
