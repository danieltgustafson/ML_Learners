import numpy as np
import scipy.stats as sp
#import pandas as pd

class RTLearner(object):

	def __init__(self,leaf_size=1, verbose = False):
			self.leaf_size=leaf_size
			# move along, these aren't the drones you're looking for

	def addEvidence(self,x,y):
		#df=pd.DataFrame(x)
		#df['y']=y
		#print len(set(y))
		#if len(set(y))<=10:
		#	self.fun=sp.mode
			#print 'no'
		#else:
		#self.fun=np.mean
			#print 'lame'

		if len(y.shape)==1:
			y=np.array([y]).transpose()
		elif len(y.shape)==2 and y.shape[0]!=x.shape[0]:
			y=y.transpose()
		df=np.append(x,y,1)
		self.build_tree(df)
		
	def build_tree(self,data):

		if data.shape[0]<=self.leaf_size: 
			val=np.mean(data[:,-1])
			#val=self.fun(data.iloc[:,-1])
			#try:
			#	return [-99,val[0][0],-99,-99]
			#except:
			#	return[-99,val,-99,-99]
			return[-99,val,-99,-99]
		if data[:,-1].max()==data[:,-1].min():
			return [-99,data[:,-1].min(),-99,-99]

		else:
			split_feat = np.random.randint(0,(data.shape[1]-1))
			rand_rows=np.random.choice(data.shape[0],size=2,replace=False)
			count=0
			while data[rand_rows[0],split_feat]==data[rand_rows[1],split_feat] and count<10:
				rand_rows=np.random.choice(data.shape[0],size=2,replace=False)
				count+=1
			split_val  = (data[rand_rows[0],split_feat]+\
			data[rand_rows[1],split_feat])/2
			if (count==10 and data[data[:,split_feat] >  split_val].shape[0]==0):
				val=np.mean(data[:,-1])
				return [-99,val,-99,-99]
				#try:
				#	return [-99,val[0][0],-99,-99]
				#except:
				#	return[-99,val,-99,-99]
			left = self.build_tree(data[data[:,split_feat] <= split_val])
			right = self.build_tree(data[data[:,split_feat] >  split_val])
			if len(np.shape(left))>1:
				length=np.shape(left)[0]
			else:
				length=1
			root=[np.int_(split_feat),split_val,1,np.int_(length+1)]
			self.tree=np.vstack((root,left,right))
			return self.tree

		
	def query(self,q):
		answer=[]
		for i in q:
			answer.append(self.query_one(i))
		return answer

	def author(self):
		return 'dgustafson6'
	
	def query_one(self,q,tree=None):
		if tree==None:
			tree=self.tree
		if tree[0][0]==-99:
			return tree[0][1]
		elif q[tree[0][0]]<=tree[0][1]:
			return self.query_one(q,tree[range(int(tree[0][2]),len(tree)),:])
		else:
			return self.query_one(q,tree[range(int(tree[0][3]),len(tree)),:])



