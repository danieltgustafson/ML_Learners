Usage:
python bestproj2.py new|load

new = retrain the neural network from scratch
load = load the existing model weights (target_q.pkl +action_q.pkl)

Requires:
gym
tempfile
numpy as np 
gym.wrappers
sklearn
sklearn.utils.shuffle
sklearn.linear_model.LinearRegression
sklearn.neural_network.MLPRegressor
sklearn.externals.joblib
random
sklearn.preprocessing
time
pickle
warnings
sys

