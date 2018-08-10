

import numpy as np
import scipy.stats as sp
#import pandas as pd
import RTLearner as rt
class BagLearner(object):

	def __init__(self,learner=rt.RTLearner,kwargs = {'leaf_size':1},bags=20,boost=False, verbose = False):
			self.args=kwargs
			self.bags=bags
			self.boost=boost
			self.learner=learner
			pass
			# move along, these aren't the drones you're looking for

	def fit(self,x,y): 
		self.learner_list=[]
		if self.bags>0:
			for i in range(0,self.bags):
				vals=np.random.choice(len(y),size=len(y),replace=True)
				self.learner_list.append(self.learner(self.args.values()[0],verbose=False))
				self.learner_list[i].addEvidence(x[vals],y[vals])
		else:
			self.learner_list.append(self.learner(self.args.values()[0],verbose=False))
			self.learner_list[0].addEvidence(x,y)

		
	def predict(self,q):
		answer=[]
		for i in range(0,len(self.learner_list)):
			answer.append(self.learner_list[i].query(q))
		answer=np.array(answer).mean(axis=0)
		return answer
  
	def author(self):
		return 'dgustafson6'
