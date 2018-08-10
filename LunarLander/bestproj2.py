
import gym
import tempfile
import numpy as np 
from gym import wrappers
import sklearn
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import random
from sklearn import preprocessing
import time
import pickle
import warnings
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning) 


def gather_training(env,iterations):
	train_x=[]
	train_y=[]
	for i in range(iterations):

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
	

	sx=scalerX.transform(x)
	sy=scalerY.transform(y)

	mod=MLPRegressor(hidden_layer_sizes=(100,100),max_iter=100,tol=.01,early_stopping=True,warm_start=True)
	mod.fit(x,y)
	return mod#,scalerX,scalerY


def q_learning(env, init_model, iterations=10000,alpha=.01,gamma=.99925,rar=.9999,radr=.999997):

	action_list=np.array(range(4))
	action_list.shape=(4,1)
	new_x=[]
	new_y=[]
	target_x=[]
	target_y=[]
	reward_dict={}
	#delta=[]
	prime_model=MLPRegressor(hidden_layer_sizes=(100,25,25,25),max_iter=200,tol=.001,learning_rate_init=alpha,warm_start=True,solver='adam')#,learning_rate='adaptive')#,learning_rate='adaptive',
	        #learning_rate_init=.01)
	target_model=MLPRegressor(hidden_layer_sizes=(100,25,25,25),max_iter=200,tol=.001,learning_rate_init=alpha,warm_start=True,solver='adam')
	for its in range(iterations):
	        print its,rar
	        reward_track=0
	        state=env.reset()

	        if its>50:
	                model=prime_model
	        else:
	                model=init_model
	        if its>50:
	                target_mod=target_model
	        else:
	                target_mod=init_model

	        #start=time.time()
	        #state=np.append(state,0)
	        counts=0
	        state=np.append(state,counts)
	        while True:

	                values=[]
	                values=np.hstack((np.tile(state,(4,1)),action_list))
	                if (np.random.random()<rar) and (its <24500) and (its+1)%500!=0:

	                        action=env.env.action_space.sample()
	                        values=values[action]
	                        pred_val=model.predict(values)[0]
	                else:

	                        preds=model.predict(values)
	                        action=np.argmax(preds)
	                        pred_val=preds[action]
	                        values=values[action]


	                new_state,reward,done,info = env.step(action)
	                new_state=np.append(new_state,counts)


	                new_vals=[]
	                new_vals=np.hstack((np.tile(new_state,(4,1)),action_list))

	                pred_prime=target_mod.predict(new_vals)
	                Q_action=model.predict(new_vals).argmax()


	                #added the min pred prime/200 to prevent blowing up beyond reasonable value states
	                new_x.append(values)

	                new_y.append(reward+(gamma*pred_prime.max()*(not done)))#-counts/float(500))
	                target_x.append(values)
	                target_y.append(reward+gamma*(not done)*pred_prime[Q_action])#-counts/float(500))


	                state=new_state
	                rar=rar*radr
	                reward_track+=reward


	                if done:
	                        print reward_track, 'counts =', counts
	                        reward_dict[its]=reward_track
	                        break

	        if len(target_y)>100000 or its==50:
	                

	                ty=np.array(target_y)
	                tx=np.array(target_x)

	                t_s_x,t_s_y=zip(*random.sample(zip(target_x,target_y),min(50000,len(target_y))))

	                target_x,target_y=list(t_s_x),list(t_s_y)
	                target_model.partial_fit(target_x,target_y)

	        if len(new_y)>100000 or its==50:
	                print len(new_x),rar
	                print its, reward_track

	                y=np.array(new_y)
	                x=np.array(new_x)
			
	                s_x,s_y=zip(*random.sample(zip(x,y),min(len(y),50000)))

	                prime_model.partial_fit(s_x,s_y)
	                ##NOTE THE BELOW IS ONLY FOR PARTIAL FITS!
	                new_x,new_y=list(s_x),list(s_y)

	return reward_dict,target_model,prime_model

def q_exist(env, prime_model,target_model, iterations=10000,alpha=.01,gamma=.99925,rar=0,radr=.999997):

	action_list=np.array(range(4))
	action_list.shape=(4,1)
	new_x=[]
	new_y=[]
	target_x=[]
	target_y=[]
	reward_dict={}

	for its in range(iterations):
	        print its,rar
	        reward_track=0
	        state=env.reset()       
	        model=prime_model 
	        target_mod=target_model
	        counts=0
	        state=np.append(state,counts)
	        while True:
	                values=[]

	                values=np.hstack((np.tile(state,(4,1)),action_list))
	                if (np.random.random()<rar) and (its <24500) and (its+1)%500!=0:
	                        #rar=rar*radr
	                        action=env.env.action_space.sample()
	                        values=values[action]
	                        pred_val=model.predict(values)[0]
	                else:

	                        preds=model.predict(values)
	                        action=np.argmax(preds)
	                        pred_val=preds[action]
	                        values=values[action]

	                new_state,reward,done,info = env.step(action)
	                new_state=np.append(new_state,counts)

	                new_vals=[]
	                new_vals=np.hstack((np.tile(new_state,(4,1)),action_list))

	                pred_prime=target_mod.predict(new_vals)
	                Q_action=model.predict(new_vals).argmax()

	                #added the min pred prime/200 to prevent blowing up beyond reasonable value states
	                new_x.append(values)
	                #new_y.append(pred_val+(alpha*(reward+(gamma*pred_prime.max()*(not done))-pred_val)))
	                new_y.append(reward+(gamma*pred_prime.max()*(not done)))#-counts/float(500))
	                target_x.append(values)
	                target_y.append(reward+gamma*(not done)*pred_prime[Q_action])#-counts/float(500))


	                state=new_state
	                rar=rar*radr
	                reward_track+=reward
	                if done:
	                        print reward_track, 'counts =', counts
	                        reward_dict[its]=reward_track
	                        break

	        if len(target_y)>100000 or its==50:
	          

	                ty=np.array(target_y)
	                tx=np.array(target_x)

	                t_s_x,t_s_y=zip(*random.sample(zip(target_x,target_y),min(50000,len(target_y))))

	                target_x,target_y=list(t_s_x),list(t_s_y)
	                target_model.partial_fit(target_x,target_y)

	        if len(new_y)>100000 or its==50:

	                y=np.array(new_y)
	                x=np.array(new_x)
	                s_x,s_y=zip(*random.sample(zip(x,y),min(len(y),50000)))

	                prime_model.partial_fit(s_x,s_y)
	                ##NOTE THE BELOW IS ONLY FOR PARTIAL FITS!
	                new_x,new_y=list(s_x),list(s_y)

	return reward_dict,target_model,prime_model
def reset():
	env.close()
	env.render(close=True)

def make():	
	tdir=tempfile.mkdtemp()
	env=gym.make('LunarLander-v2')
	env=wrappers.Monitor(env,tdir,force=True)
def upload():
	gym.upload(tdir,api_key='sk_mg3eTnr2RuOUQ1sytvKfw')

if __name__ == "__main__":

	tdir=tempfile.mkdtemp()
	env=gym.make('LunarLander-v2')
	env2=wrappers.Monitor(env,tdir,force=True)
	

	if sys.argv[-1] == 'new':
		print 'Gathering random actions...(1000 episodes ~5-10s)'
		ix,iy=gather_training(env,1000)
		print 'Training an initialization model'
		init=train_model(ix,iy)
		'Running Learner'
		reward_dict,target_model,prime_model=q_learning(env2,init)
		print tdir
	elif sys.argv[-1]=='load':
		prime = joblib.load('action_q.pkl')
		target=joblib.load('target_q.pkl')
		q_exist(env2,prime,target)





