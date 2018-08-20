#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/8/20 10:09
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  NaiveBayes
# @File      :  NavieBayes_test3.py
# @Software  :  PyCharm

# ********************************************************* 
import numpy as np
import random
import feedparser

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def bagOfWords2VecMN(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	# p0Num = np.zeros(numWords)
	# p1Num = np.zeros(numWords)
	# p0Denom = 0.0 ; p1Denom = 0.0
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	p0Denom = 2.0 ; p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# p1Vect = p1Num/p1Denom
	# p0Vect = p0Num/p0Denom
	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
	return sortedFreq[:30]

def localWords(feed1,feed0):
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList,fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	trainingSet = list(range(2*minLen))
	testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is : ',float(errorCount)/len(testSet))
	return vocabList,p0V,p1V

def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V = localWords(ny,sf)
	topNY = []
	topSF = []
	for i in range(len(p0V)):
		if p0V[i] > -6.0:
			topSF.append((vocabList[i],p0V[i]))
		if p1V[i] > -6.0:
			topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF,key=lambda pair : pair[1],reverse=True)
	print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF')
	for item in sortedSF:
		print(item[0])
	sortedNY = sorted(topNY,key=lambda pair: pair[1],reverse=True)
	print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY')
	for item in sortedNY:
		print(item[0])

if __name__ == '__main__':
	ny = feedparser.parse('http://rss.yule.sohu.com/rss/yuletoutiao.xml')
	sf = feedparser.parse('http://www.cppblog.com/kevinlynx/category/6337.html/rss')
	getTopWords(ny,sf)