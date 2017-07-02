import gym
import tempfile
import numpy as np 
from gym import wrappers
import sklearn
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import random
from sklearn import preprocessing
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#args=sys.argv[1],sys.argv[2],sys.argv[3]
tdir=tempfile.mkdtemp()
env=gym.make('LunarLander-v2')
env=wrappers.Monitor(env,tdir,force=True)

def gather_training(env,iterations):
	train_x=[]
	train_y=[]
	for i in range(iterations):
		print i
		state=env.reset()
		start=time.time()
		state=np.append(state,0)
		while True:
			action=env.env.action_space.sample()
			new_state,reward,done,info=env.step(action)
			train_x.append(np.append(state,action))
			train_y.append(reward)
			state=np.append(new_state,time.time()-start)
			if done:
				break
	return train_x,train_y

def train_model(x,y):
	y=np.array(y)
	x=np.array(x)
	scalerX=preprocessing.StandardScaler().fit(x)
	scalerY=preprocessing.StandardScaler().fit(y)
	
	#pos_indices=np.where(y>10)[0][-10000:,]
	#print(len(pos_indices))

	#if len(y)>250000:
	#pos_x,pos_y=np.repeat(x[pos_indices],5,axis=0),np.repeat(y[pos_indices],5,axis=0)
	
		#s_x,s_y=zip(*random.sample(zip(np.vstack((x[-1000000:,],pos_x)),np.append(y[-1000000:,],pos_y)),min(len(y),250000)))
	#else:
	#	s_x,s_y=zip(*random.sample(zip(x,y),len(y)))

	#s_x=scaler.transform(x)

	#s_x,s_y = shuffle(new_x,new_y,random_state=0)
	#scaler.StandardScaler()
	#scaler.fit(x)
	sx=scalerX.transform(x)
	sy=scalerY.transform(y)
	#x=np.array(x)
	#dist = np.sqrt((x[:,0])**2 + (x[:,1])**2)
	#dist.shape=(len(dist),1)
	#x=np.hstack((dist,x[:,2:]))
	#f type==RandomForestRegressor:
	#	mod=type(n_estimators=4,min_samples_split=10,n_jobs=5,max_depth=1000,bootstrap=False)
	#elif(type==SVR):
	#	mod=type(kernel='rbf')
	#else:
	#mod=type(n_jobs=1,n_estimators=5,max_depth=1000)
	mod=MLPRegressor(hidden_layer_sizes=(100,100),max_iter=100,tol=.01,early_stopping=True,warm_start=True)
	#weights=np.array([1]*len(y))
	#weights[np.where(y>0)[0]]=y[np.where(y>0)[0]]+10
	#mod.fit(x,y,sample_weight=weights)
	mod.fit(x,y)
	return mod#,scalerX,scalerY


def q_learning(env, init_model,\
	iterations=5000,alpha=.55,gamma=1,rar=.9999,radr=.999):

	action_list=np.array(range(4))
	action_list.shape=(4,1)
	new_x=[]
	new_y=[]
	target_x=[]
	target_y=[]
	#reward_dict={}
	#delta=[]
	prime_model=MLPRegressor(hidden_layer_sizes=(100,100,25),max_iter=300,tol=.001,warm_start=True,solver='adam')#,learning_rate='adaptive')#,learning_rate='adaptive',
		#learning_rate_init=.01)
	target_model=MLPRegressor(hidden_layer_sizes=(100,100,25),max_iter=300,tol=.001,warm_start=True,solver='adam')
	for its in range(iterations):
		print its,rar
		reward_track=0
		state=env.reset()

		if its>90:
			model=prime_model
		else:
			model=init_model
		if its>140:
			target_mod=target_model
		else:
			target_mod=init_model

		#start=time.time()
		#state=np.append(state,0)
		counts=0
		state=np.append(state,counts)
		while True:
			#dist=np.array([np.sqrt((state[0])**2 + (state[1])**2)]*4)
			#dist.shape=(4,1)
			values=[]
			#values=np.hstack((dist,np.tile(state[2:],(4,1)),action_list))
			values=np.hstack((np.tile(state,(4,1)),action_list))
			if (np.random.random()<rar) and (its <24500) and (its+1)%500!=0:
				#rar=rar*radr
				action=env.env.action_space.sample()
				values=values[action]
				pred_val=model.predict(values)[0]
			else:
				#values=np.hstack((dist,np.tile(state[2:],(4,1)),action_list))
				preds=model.predict(values)
				action=np.argmax(preds)
				pred_val=preds[action]
				values=values[action]


			new_state,reward,done,info = env.step(action)
			#episode_time=time.time()-start
			new_state=np.append(new_state,counts)#episode_time)
			
			new_vals=[]
			new_vals=np.hstack((np.tile(new_state,(4,1)),action_list))
			
			pred_prime=target_mod.predict(new_vals)
			Q_action=model.predict(new_vals).argmax()
			#pred_val=min(300,pred_val)
			
			#added the min pred prime/200 to prevent blowing up beyond reasonable value states
			new_x.append(values)
			#new_y.append(pred_val+(alpha*(reward+(gamma*pred_prime.max()*(not done))-pred_val)))
			new_y.append((alpha*(reward+(gamma*pred_prime.max()*(not done))))-counts/10)
			target_x.append(values)
			target_y.append(reward+gamma*(not done)*pred_prime[Q_action]-counts/10)
			

			state=new_state
			rar=rar*radr
			reward_track+=reward
			#alpha=.99999999*alpha

			if done:
				print reward_track
				#reward_dict[its]=reward_track
				break

		if (its+1)%133==0:
			print "TEST ME FUCKING FUCKFUCK"

			target_model.partial_fit(target_x,target_y)

			t_s_x,t_s_y=zip(*random.sample(zip(target_x,target_y),min(1000000,len(target_y)/2)))

			target_x,target_y=list(t_s_x),list(t_s_y)
		if (its+1)%88==0:
			print len(new_x),rar
			print its, reward_track
			#new_x,new_y=new_x[-5000000:],new_y[-5000000:]
			y=np.array(new_y)
			x=np.array(new_x)
			#pos_indices=np.where((y>np.mean(y[-500000:])+2*np.std(y[-500000:])))[0]
			
			
			ordered=np.argsort(y)[-(len(y)/10):,]
			
			pos_x,pos_y=np.repeat(x[ordered],5,axis=0),np.repeat(y[ordered],5,axis=0)
			pos=np.where(pos_y>0)[0]
			pos_x,pos_y=pos_x[pos],pos_y[pos]

			print len(pos)
			#s_x,s_y=zip(*random.sample(zip(np.vstack((x[-1500000:,],pos_x[-1500000:,])),np.append(y[-1500000:,],pos_y[-1500000:,])),min(len(y),500000)))
			s_x,s_y=zip(*random.sample(zip(np.vstack((x,pos_x)),np.append(y,pos_y)),min(len(y),1000000)))
			prime_model.partial_fit(s_x,s_y)
			#new_x,new_y=zip(*random.sample(zip(new_x,new_y),min(len(new_y),1000000)))
			#new_x=list(new_x)
			#new_y=list(new_y)
			#new_x=nex_x[-500000:,]
			#model=train_model(new_x,new_y,type)
			##NOTE THE BELOW IS ONLY FOR PARTIAL FITS!
			new_x,new_y=list(s_x),list(s_y)

	return target_model,prime_model

def reset():
	env.close()
	env.render(close=True)

def make():	
	tdir=tempfile.mkdtemp()
	env=gym.make('LunarLander-v2')
	env=wrappers.Monitor(env,tdir,force=True)

gym.upload(tdir,api_key='sk_mg3eTnr2RuOUQ1sytvKfw')
pi,Q=q_learning(env)
print(pi,Q)
env.close()

