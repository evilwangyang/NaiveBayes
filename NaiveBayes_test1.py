#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/8/6 21:26
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  NaiveBayes
# @File      :  NaiveBayes_test1.py
# @Software  :  PyCharm Community Edition

# ********************************************************* 
import numpy as np

def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
	               ['maybe','not','take','him','to','dog','park','stupid'],
	               ['my','dalmation','is','so','cute','I','love','him'],
	               ['stop','posting','stupid','worthless','garbage'],
	               ['mr','licks','ate','my','steak','how','to','stop','him'],
	               ['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]  #1代表侮辱性文字，0代表正常言论
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print('the word: %s is not in my Vocabulary!' % word)
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = np.zeros(numWords)
	p1Num = np.zeros(numWords)
	p0Denom = 0.0 ; p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive

if __name__ == '__main__':
	listOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	print(myVocabList)
	returnVec1 = setOfWords2Vec(myVocabList,listOPosts[0])
	print(returnVec1)
	returnVec2 = setOfWords2Vec(myVocabList,listOPosts[3])
	print(returnVec2)