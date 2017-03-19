

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data




if __name__=="__main__":
    if len(sys.argv) != 4:
        print "Usage: python testlearner.py <filename> <test_type(rt/lin)>"
        sys.exit(1)
    inf = open(sys.argv[1])
    try:
        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    except:
        inf = open(sys.argv[1])
        data = np.array([map(float,s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
        

    
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows
    print data.shape

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    start = time.time()


    if sys.argv[2]=='rt':
        rmse_in=[]
        rmse_out=[]
        for i in range(1,25):
            print 'trial', i
            rmse_a=[]
            rmse_b=[]
            for j in range(1,int(sys.argv[3])):
                learner = rt.RTLearner(verbose = True,leaf_size=i) 
                learner.addEvidence(trainX, trainY)
            # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                check_in=[]
                check_out=[]
                #for i in range(0,len(trainY)):
                 #   if trainY[i]==predY[i]:
                  #      check_in.append(1)
                   # else:
                    #    check_in.append(0)
               # correct_rate_in = 100*(sum(check_in))/trainY.shape[0]
                rmse_a.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
                print 'leaf',i,'trial',j
                #print "In sample results"
                #print "RMSE: ", rmse_in[i]
                #print "Correct%: ", correct_rate_in,'%'
                #c = np.corrcoef(predY, y=trainY)
                #print "corr: ", c[0,1]

            # evaluate out of sample
                predY = learner.query(testX)
                #for i in range(0,len(testY)):
                 #   if testY[i]==predY[i]:
                  #      check_out.append(1)
                   # else:
                    #    check_out.append(0)
                #correct_rate_out = 100*(sum(check_out))/trainY.shape[0]
                rmse_b.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
            print 'avg in ', np.mean(rmse_a) , 'avg out ',np.mean(rmse_b)
            rmse_in.append(np.mean(rmse_a))
            rmse_out.append(np.mean(rmse_b))
               # print "Out of sample results"
            #print "RMSE: ", rmse_out[i]
            #print "Correct%: ", correct_rate_out,'%'
            #c = np.corrcoef(predY, y=testY)
            #rint "corr: ", c[0,1]
        df_temp = pd.DataFrame(rmse_in)
        df_temp['RMSE Out Sample']=rmse_out
        df_temp.columns=['RMSE In Sample','RMSE Out Sample']
        df_temp.plot()
        plt.show()
    elif sys.argv[2]=='lin':
        learner = lrl.LinRegLearner(verbose = True) 
        learner.addEvidence(trainX, trainY) # train it
    # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]

    elif sys.argv[2]=='bl':
        rmse_in=[]
        rmse_out=[]
        for i in range(0,10):
            print 'trial', i
            rmse_a=[]
            rmse_b=[]
            for j in range(1,int(sys.argv[3])):

                learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":2}, bags = i, boost = False, verbose = False)
                learner.addEvidence(trainX, trainY)
            # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                check_in=[]
                check_out=[]
                #for i in range(0,len(trainY)):
                 #   if trainY[i]==predY[i]:
                  #      check_in.append(1)
                   # else:
                    #    check_in.append(0)
               # correct_rate_in = 100*(sum(check_in))/trainY.shape[0]
                rmse_a.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
                print 'leaf',i,'trial',j
                #print "In sample results"
                #print "RMSE: ", rmse_in[i]
                #print "Correct%: ", correct_rate_in,'%'
                #c = np.corrcoef(predY, y=trainY)
                #print "corr: ", c[0,1]

            # evaluate out of sample
                predY = learner.query(testX)
                #for i in range(0,len(testY)):
                 #   if testY[i]==predY[i]:
                  #      check_out.append(1)
                   # else:
                    #    check_out.append(0)
                #correct_rate_out = 100*(sum(check_out))/trainY.shape[0]
                rmse_b.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
            print 'avg in ', np.mean(rmse_a) , 'avg out ',np.mean(rmse_b)
            rmse_in.append(np.mean(rmse_a))
            rmse_out.append(np.mean(rmse_b))
               # print "Out of sample results"
            #print "RMSE: ", rmse_out[i]
            #print "Correct%: ", correct_rate_out,'%'
            #c = np.corrcoef(predY, y=testY)
            #rint "corr: ", c[0,1]
    elif sys.argv[2]=='bl1':
        learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":2}, bags = 20, boost = False, verbose = False)
        learner.addEvidence(trainX, trainY)
            # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse_in=math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print "In sample results"
        print "RMSE: ", rmse_in
                #print "Correct%: ", correct_rate_in,'%'
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

            # evaluate out of sample
        predY = learner.query(testX)
                #for i in range(0,len(testY)):
                 #   if testY[i]==predY[i]:
                  #      check_out.append(1)
                   # else:
                    #    check_out.append(0)
                #correct_rate_out = 100*(sum(check_out))/trainY.shape[0]
        rmse_out=math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            
        print "Out of sample results"
        print "RMSE: ", rmse_out
            #print "Correct%: ", correct_rate_out,'%'
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]

    elif sys.argv[2]=='rt1':
        learner = rt.RTLearner(verbose = True,leaf_size=1)
        learner.addEvidence(trainX, trainY)
            # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse_in=math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print "In sample results"
        print "RMSE: ", rmse_in
                #print "Correct%: ", correct_rate_in,'%'
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

            # evaluate out of sample
        predY = learner.query(testX)
                #for i in range(0,len(testY)):
                 #   if testY[i]==predY[i]:
                  #      check_out.append(1)
                   # else:
                    #    check_out.append(0)
                #correct_rate_out = 100*(sum(check_out))/trainY.shape[0]
        rmse_out=math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            
        print "Out of sample results"
        print "RMSE: ", rmse_out
            #print "Correct%: ", correct_rate_out,'%'
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        
    end = time.time()
    print (end - start)
