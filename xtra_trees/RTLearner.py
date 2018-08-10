import numpy as np
import scipy.stats as sp
#import pandas as pd

class RTLearner(object):
	#accepts the leaf size as input - defaults to 1 (one node per observation).  Will overfit by default
	def __init__(self,leaf_size=1, max_depth=1000,verbose = False):
			self.leaf_size=leaf_size
			self.max_depth=max_depth
			
	#Take the input data and ensure it is in expected form.  Pass it to the tree building method
	def addEvidence(self,x,y):
	
		if len(y.shape)==1:
			y=np.array([y]).transpose()
		elif len(y.shape)==2 and y.shape[0]!=x.shape[0]:
			y=y.transpose()
		df=np.append(x,y,1)
		self.build_tree(df)
	
	#Build the decision Tree	
	def build_tree(self,data):
		#This is a recursive function - first check for termination conditions
		#Terminate if the length of input is <= to the leaf size parameter (default 1). Take the mean of the Y values 
		#to be the predicted value for this leaf node
		if data.shape[0]<=self.leaf_size: 
			val=np.mean(data[:,-1])
			return[-99,val,-99,-99]
		#Other termination condition - if all the remaining observations have the same Y value then end.  Predicted value
		#equals whatever the Y is.
		if data[:,-1].max()==data[:,-1].min():
			return [-99,data[:,-1].min(),-99,-99]

		else:
			#Pick a random X value (feature) to split your data on
			split_feat = np.random.randint(0,(data.shape[1]-1))
			#Pick a 2 random points (row number) to determine a split threshold
			rand_rows=np.random.choice(data.shape[0],size=2,replace=False)
			count=0
			#Check to make sure that the random row cut point doesn't result in split where one side has 0 rows
			#This can occur if by random chance, the two random rows have the same X value. 
			#Pick a new value if this occurs.  If it occurs more than 10 times in a row - give up-all rows may have
			# the same value!
			while data[rand_rows[0],split_feat]==data[rand_rows[1],split_feat] and count<10:
				rand_rows=np.random.choice(data.shape[0],size=2,replace=False)
				count+=1
			#split threshold is just the mean of the two random rows we chose.
			split_val  = (data[rand_rows[0],split_feat]+\
			data[rand_rows[1],split_feat])/2
			
			#If we hit our loop limit and the row values are still equal - assume all input values are the same
			#This would be a USELESS input feature to try to create a branch - just return a leaf node= mean of Y!
			if (count==10 and data[data[:,split_feat] >  split_val].shape[0]==0):
				val=np.mean(data[:,-1])
				return [-99,val,-99,-99]
			#Call the recursion - left branch is <= the split val, right sends the data >
			left = self.build_tree(data[data[:,split_feat] <= split_val])
			right = self.build_tree(data[data[:,split_feat] >  split_val])
			if len(np.shape(left))>1:
				length=np.shape(left)[0]
			else:
				length=1
			#fork pointing to the left and right. First value indicates the feature, 2nd is split threshold
			#Last two values point to where the next node lies (Left node will always be 1 below, right will be below
			#ALL of the left branches. 
			root=[np.int_(split_feat),split_val,1,np.int_(length+1)]
			self.tree=np.vstack((root,left,right))
			return self.tree

	# Query the tree -dataset query - iterate over a single value query	
	def query(self,q):
		answer=[]
		for i in q:
			answer.append(self.query_one(i))
		return answer

	def author(self):
		return 'dgustafson6'
	
	#Query a single value - trace the tree data map and find the predicted answer
	def query_one(self,q,tree=None):
		#if tree==None:
		tree=self.tree
		if tree[0][0]==-99:
			return tree[0][1]
		elif q[int(tree[0][0])]<=tree[0][1]:
			return self.query_one(q,tree[range(int(tree[0][2]),len(tree)),:])
		else:
			return self.query_one(q,tree[range(int(tree[0][3]),len(tree)),:])



