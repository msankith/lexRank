import get_tokens
from math import log10
import numpy as np
import operator

class lexranker:
	threshold = 0.1
	e = 0.0
	allsentences = []
	wordset = []
	wordidf = {}
	markov = []
	def __init__(self,documents):
		self.readalldocs(documents)
		self.calcwordidf()
	
	def calcwordidf(self):
		N = len(self.allsentences)
		
		for w in self.wordset:
			word_in_num_sentences = 0
			for s in self.allsentences:
				sentencewords = get_tokens.get_words_from_sentence(s)
				if w in sentencewords:
					word_in_num_sentences+=1
			
			self.wordidf[w] = log10(float ( (N/word_in_num_sentences) ) )
			
	def readalldocs(self,documents):
		docindex = 1
		for doc in documents:
			fh = open(doc,'r')
			doccontent = fh.read().strip()
			sentences = get_tokens.get_sentences_from_text(doccontent)
			for s in sentences:
				self.allsentences.append(s)
				words = get_tokens.get_words_from_sentence(s)
				for word in words:
					self.wordset.append(word)
					'''
					if word not in wordidf:
						wordidf[word] = 1
					else:
						if word in oldwordidf:
							if oldwordidf[word] == wordidf[word]:
								wordidf[word]+=1
					'''
			docindex+=1
		
		self.allsentences = list(set(self.allsentences))
		self.wordset = list(set(self.wordset))
		self.cosinematrix = {s1:{s2:0 for s2 in self.allsentences} for s1 in self.allsentences}

		'''
		N = len(documents)
		for key,value in wordidf.iteritems():
			wordidf[key] = log10(float((N/wordidf[key])+1.0))
		'''

	def idf_modified_cosine(self,s1,s2):
		words1 = get_tokens.get_words_from_sentence(s1)
		words2 = get_tokens.get_words_from_sentence(s2)
		words = list(words1+words2)
		mag1 = 0.0
		mag2 = 0.0
		dotprod = 0.0

		for w in words1:
			tf = words1.count(w)
			idf = self.wordidf[w]
			mag1 += (tf*idf)**2
		mag1 = float(mag1)**0.5

		for w in words2:
			tf = words2.count(w)
			idf = self.wordidf[w]
			mag2 += (tf*idf)**2
		mag2 = float(mag2)**0.5

		for w in words:
			tf1 = words1.count(w)
			tf2 = words2.count(w)
			idf = self.wordidf[w]
				
			dotprod += (tf1*tf2*(idf**2))

		return (dotprod*1.0)/(mag1*mag2*2)

	def diffmagnitude(self,v1,v2):
		diffvec = []
		for i in range(len(v1)):
			diffvec.append(v1[i]-v2[i])
		mag = 0.0
		
		for v in diffvec:
			mag += v**2

		return mag**0.5
		
	def init_sent_vector(self):
		self.first = []
		for s in self.allsentences:
			self.first.append(1.0/len(self.allsentences))
		self.markov.append(self.first)
		self.print_matrix(self.markov,"markov_process.csv")
	
	def stationarydistribution(self,matrix,n):
		matrix = np.array(matrix)
		matrixtrans = matrix.transpose()

		np.dot(matrixtrans,self.first)
		pt_1 = list(np.dot(matrixtrans,self.first))
		self.markov.append(pt_1)
		pt = list(self.first)
		while self.diffmagnitude(pt,pt_1) > self.e:
			pt = list(pt_1)
			pt_1 = list(np.dot(matrixtrans,pt))
			self.markov.append(pt_1)
		self.print_matrix(self.markov,"markov_process.csv")
		return pt_1

	def TwoDDictToMatrix(self,thisdict):
		matrix = []
		for k1,v in thisdict.iteritems():
			row = []
			for k2, v2 in v.iteritems():
				row.append(v2)
			matrix.append(row)
		return matrix

	def calc_cosine_matrix(self):
		self.cosinematrix = {s1:{s2:0 for s2 in self.allsentences} for s1 in self.allsentences}
		
		for s1 in self.allsentences:
			for s2 in self.allsentences:
				self.cosinematrix[s1][s2] = self.idf_modified_cosine(s1,s2)

		self.actualcosinematrix = self.TwoDDictToMatrix(self.cosinematrix)
		self.print_matrix(self.actualcosinematrix,"cosine_similarity_matrix.csv")

	def drop_edges(self):
		self.degree = {s:0 for s in self.allsentences}
		for s1 in self.allsentences:
			for s2 in self.allsentences:
				self.cosinematrix[s1][s2] = self.idf_modified_cosine(s1,s2)
				if self.cosinematrix[s1][s2] > self.threshold:
					self.cosinematrix[s1][s2] = 1
					self.degree[s1]+=1
				else:
					self.cosinematrix[s1][s2] = 0

		self.actualcosinematrix = self.TwoDDictToMatrix(self.cosinematrix)
		self.print_matrix(self.actualcosinematrix,"matrix_before_normalization.csv")

	def normalize_matrix(self):
		for s1 in self.allsentences:
			for s2 in self.allsentences:
				if self.degree[s1]!=0:
					self.cosinematrix[s1][s2] = self.cosinematrix[s1][s2]/(self.degree[s1]*1.0)

		self.actualcosinematrix = self.TwoDDictToMatrix(self.cosinematrix)
		self.print_matrix(self.actualcosinematrix,"stochastic_matrix.csv")

	def calc_sent_scores(self):
		eigenvector = []
		sentenceeigen = {}
		
		eigenvector = self.stationarydistribution(self.actualcosinematrix,len(self.allsentences))

		indexeigen = 0
		for s in self.allsentences:
			sentenceeigen[s] = eigenvector[indexeigen]
			indexeigen+=1

		sentenceeigen = sorted(sentenceeigen.keys(),key=lambda x:sentenceeigen[x],reverse=True)

		return sentenceeigen
		
	def lex_rank(self):
		self.calc_cosine_matrix()
		self.drop_edges()
		self.normalize_matrix()	

		self.init_sent_vector()
		sentenceeigen = self.calc_sent_scores()
		return sentenceeigen

	def print_matrix(self,matrix,filename):
		fh = open(filename,"w+")

		for row in matrix:
			fh.write("\n")
			for column in row:
				fh.write(str(column)+",")
