"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

	def __init__(self, \
		num_states=100, \
		num_actions = 4, \
		alpha = 0.2, \
		gamma = 0.9, \
		rar = 0.5, \
		radr = 0.99, \
		dyna = 0, \
		verbose = False,\
		base=10):

		self.base=base
		self.dyna=dyna
		self.rar=rar
		self.radr=radr
		self.verbose = verbose
		self.num_actions = num_actions
		self.num_states=num_states
		self.s = 0
		self.a = 0
		self.alpha=alpha
		self.gamma=gamma
		self.T=np.array([-1,-1,-1])
		self.R=np.zeros((num_states,num_actions))
		#for i in num_actions:
		#	rand=np.random.uniform(-1,1,num_states)
		self.Q=np.zeros((num_states,num_actions))

	def querysetstate(self, s):
		"""
		@summary: Update the state without updating the Q-table
		@param s: The new state
		@returns: The selected action
		"""
		self.s = int(s[0])*int(s[1:],self.base)
		self.action = rand.randint(0, self.num_actions-1)
		if self.verbose: print "s =", self.s,"a =",self.action
		return self.action

	def answer(self,s):
		s=int(s[0])*int(s[1:],self.base)
		unique_actions=list(np.unique(self.Q[s]))
		if (len(unique_actions)==1) or (unique_actions==[-10,0]):
			action=1
		else:
			action=self.Q[s].argmax()

		return action 
	def query(self,s_prime,r):
		"""
		@summary: Update the Q table and return an action
		@param s_prime: The new state
		@param r: The ne state
		@returns: The selected action
		"""
		s_prime=int(s_prime[0])*int(s_prime[1:],self.base)
		if self.verbose: 
			print 's_old=',self.s,"s =", s_prime,"a =",self.action,"r =",r
			print (1-self.alpha)*self.Q[self.s,self.action]+self.alpha*(r+self.gamma*self.Q[s_prime].max())
		if self.dyna>0:
			self.T=np.vstack([self.T,[self.s,self.action,s_prime]])
			self.R[self.s,self.action]=(1-self.alpha)*self.R[self.s,self.action]+self.alpha*r
			if len(self.T)%5==0:
				for i in range(0,self.dyna):
					self.halucinate()

		self.Q[self.s,self.action]=(1-self.alpha)*self.Q[self.s,self.action]+self.alpha*\
		(r+self.gamma*self.Q[s_prime].max())

		chance_card=np.random.binomial(1,self.rar)
		unique_actions=list(np.unique(self.Q[s_prime]))

		if (chance_card==1) or (len(unique_actions)==1) or (unique_actions==[-10,0]):
		    self.action = rand.randint(0, self.num_actions-1)
		    #print 'rand', self.action
		else:
		    self.action = self.Q[s_prime].argmax()
		    #print 'non-rand', self.action
		#if self.verbose: print 's_old=',self.s,"s =", s_prime,"a =",self.action,"r =",r
		self.s=s_prime
		self.rar=self.rar*self.radr
		#print self.Q[0:5]
		return self.action

	def halucinate(self):
		count=0
		a = rand.randint(0, self.num_actions-1)
		s = rand.randint(0, self.num_states-1)
		#s = np.base_repr(s,self.base)
		T_lim=self.T[(self.T[:,0]==s)&(self.T[:,1]==a)]
		while len(T_lim)==0 & count< 5:
			a = rand.randint(0, self.num_actions-1)
			s = rand.randint(0, self.num_states-1)
			#s = np.base_repr(s,self.base)
			T_lim=self.T[(self.T[:,0]==s)&(self.T[:,1]==a)]
			count+=1
		if len(T_lim)>0:
			Tc=np.unique(T_lim[:,2],return_counts=True)
			T_prob=Tc[1]/float(len(T_lim))
			T_index=np.random.multinomial(1,T_prob)
				
			#else:
			#	T_index=np.random.multinomial(1,[1/len()])
			s_prime = Tc[0][T_index.argmax()]
			r=self.R[s,a]
			self.Q[s,a]=(1-self.alpha)*self.Q[s,a]+self.alpha*\
			(r+self.gamma*self.Q[s_prime].max())
	
	def author(self):
		return 'dgustafson6'



	if __name__=="__main__":
		print "Remember Q from Star Trek? Well, this isn't him"

