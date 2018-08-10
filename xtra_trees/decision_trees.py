import numpy as np
from collections import Counter

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function,class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)

def build_decision_tree():
    """Create decision tree
    capable of handling the provided 
    data."""
    # TODO: build full tree from root
    decision_tree_root = None
    root_function=lambda feature : feature[0] == 1
    node_1_function=lambda feature : feature[2]==feature[3]

    
    decision_tree_root_left = DecisionNode(None, None, None, 1)
    decision_tree_root_right_left=DecisionNode(None,None,None,1)
    decision_tree_root_right_right=DecisionNode(None,None,None,0)
    decision_tree_root_right = DecisionNode(decision_tree_root_right_left, decision_tree_root_right_right, node_1_function)
    decision_tree_root=DecisionNode(decision_tree_root_left,decision_tree_root_right,root_function)
    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    #TODO output should be [[true_positive, false_negative], [false_positive, true_negative]]
    true_positive=0
    false_negative=0
    false_positive=0
    true_negative=0
    for i in range(len(classifier_output)):
        if classifier_output[i]==1:
            if classifier_output[i] == true_labels[i]:
                true_positive+=1
            else:
                false_positive+=1
        else:
            if classifier_output[i] == true_labels[i]:
                true_negative+=1
            else:
                false_negative+=1
    
    return [[true_positive, false_negative], [false_positive, true_negative]]


def precision(classifier_output, true_labels):
    #TODO precision is measured as: true_positive/ (true_positive + false_positive)
    matrix=confusion_matrix(classifier_output,true_labels)
    precision=matrix[0][0]/float(matrix[0][0]+matrix[1][0])
    return precision
    
def recall(classifier_output, true_labels):
    #TODO: recall is measured as: true_positive/ (true_positive + false_negative)
    matrix=confusion_matrix(classifier_output,true_labels)
    recall=matrix[0][0]/float(matrix[0][0]+matrix[0][1])
    return recall
    
def accuracy(classifier_output, true_labels):
    #TODO accuracy is measured as:  correct_classifications / total_number_examples
    matrix=confusion_matrix(classifier_output,true_labels)
    accuracy=(matrix[0][0]+matrix[1][1])/float(sum(matrix[0])+sum(matrix[1]))
    return accuracy

def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    # TODO: finish this
    total=len(class_vector)
    success=sum(class_vector)
    fail=total-success
    #print total,success,fail
    
    if(success>0 and fail>0):
        success_prob=success/float(total)
        fail_prob=fail/float(total)
    else:
        success_prob=1
        fail_prob=1
    entropy = -((success_prob*np.log2(success_prob))+(fail_prob*np.log2(fail_prob)))

    
    return entropy
    
    
def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (a list of 
    lists where each list has 0 and 1 values)."""
    # TODO: finish this
    if(any(isinstance(i, list) for i in previous_classes)):
        prev_total= sum([len(i) for i in previous_classes])
        expected_prev= sum([(len(i)/float(prev_total))*entropy(i) for i in previous_classes])

    else:
        prev_total=len(previous_classes)
        expected_prev = entropy(previous_classes)
    
    
    if(any(isinstance(i, list) for i in current_classes)):
        cur_total = sum([len(i) for i in current_classes])
        expected_cur = sum([(len(i)/float(cur_total))*entropy(i) for i in current_classes])
        
    else:
        cur_total=len(current_classes)
        expected_cur = entropy(current_classes)
    
 
   
    
    IG=expected_prev - expected_cur
    
    return IG

class DecisionTree():
    """Class for automatic tree-building
    and classification."""
    
    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__()."""
        self.root = self.__build_tree__(features, classes)
    
    def simulated(self,feature,classes):
        
        threshold=np.mean(feature)
        master_index=range(len(classes))
        low_class=[]
        high_class=[]
        for i in master_index:
            if(feature[i]<threshold):
                low_class.append(classes[i])
            else:
                high_class.append(classes[i])
        gain=information_gain(classes,[low_class,high_class])
        temp=10
        j=0
        while temp>0.001:
            new_threshold=feature[np.random.randint(0,len(feature))]
            low_class=[]
            high_class=[]
            for i in master_index:
                if(feature[i]<new_threshold):
                    low_class.append(classes[i])
                else:
                    high_class.append(classes[i])

            new_ig=information_gain(classes,[low_class,high_class])
            prob=min(1,np.exp((new_ig-gain)/float(temp)))
            test=np.random.choice(2,1,p=[1-prob,prob])[0]
            if(test==1):
                threshold=new_threshold
                gain=new_ig
                if(len(low_class)>0):
                    low_freq=sum(low_class)/float(len(low_class)) 
                else:
                    low_freq=0
                if(len(high_class)>0):
                    high_freq=sum(high_class)/float(len(high_class))
                else: 
                    high_freq=0
                left=False
                if(high_freq>low_freq):
                    left=True
                    
            temp=.99**j*10
            j+=1
        return [gain,threshold,left]
    def __build_tree__(self,features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        #TODO: finish this
        #print depth
        #max_depth=100
        #print depth
    
        if depth>=self.depth_limit:
            if(sum(classes)/float(len(classes))<0.5):
                leaf = DecisionNode(None,None,None,0) 
            else:
                leaf = DecisionNode(None,None,None,1)
            return(leaf)
        
        if sum(classes)==len(classes) or sum(classes)==0:
            if(max(classes)==0):
                leaf = DecisionNode(None,None,None,0)
            else:
                leaf = DecisionNode(None,None,None,1)
            return(leaf)

        else:
            max_feature=[0,0,0,0]
            j=0
            for i in np.array(features).T:
                gain,threshold,left=self.simulated(i,classes)
                if(gain>max_feature[1]):
                    max_feature=[j,gain,threshold,left]
                j+=1
            if(max_feature[3]):

                function = lambda feature : feature[max_feature[0]]>=max_feature[2]
            else:
                function = lambda feature : feature[max_feature[0]]<max_feature[2]
            depth=depth+1
            
            temp_left=(DecisionNode(None,None,None,1))
            temp_right=(DecisionNode(None,None,None,0))
            node_temp=(DecisionNode(temp_left,temp_right,function))
 
            output=np.array([node_temp.decide(feature) for feature in features])

            features=np.array(features)
            classes=np.array(classes)
            features_left=features[output==1,:].tolist()
            features_right=features[output==0,:].tolist()
            classes_left=classes[output==1].tolist()
            classes_right=classes[output==0].tolist()
            
            #DecisionNode(build_tree(features[subclass],classes[subclass],depth+=1),build_tree(features[subclass],classes[subclass],depth+=1),function)
        return(DecisionNode(self.__build_tree__(features_left,classes_left,depth),self.__build_tree__(features_right,classes_right,depth),function))   
    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = []
        #TODO: finish this
        class_labels=[self.root.decide(example) for example in features]
        return class_labels

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r ])
    classes= map(int,  out[:, class_index])
    features = out[:, :class_index]
    return features, classes

def generate_k_folds(dataset, k):
    #TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    length=len(dataset[1])
    k_assignment=[]
    features=np.array(dataset[0])
    classes=np.array(dataset[1])
    test_id=np.random.randint(0,k)
    training=[]
    test=[]
    folds=[]
    for i in range(length):
        k_assignment=np.append(k_assignment,np.random.randint(0,k))
    for i in range(k):
        training=(features[k_assignment<>i,:].tolist(),classes[k_assignment<>i].tolist())
        test=((features[k_assignment==i,:].tolist(),classes[k_assignment==i].tolist()))
        folds.append((training,test))
    return folds

class RandomForest():
    """Class for random forest
    classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        """Create a random forest with a fixed 
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of 
        decision trees."""
        # TODO implement the above algorithm
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""
        # TODO implement classification for a random forest.
        raise NotImplemented()

class ChallengeClassifier():
    
    def __init__(self):
        # initialize whatever parameters you may need here-
        # this method will be called without parameters 
        # so if you add any to make parameter sweeps easier, provide defaults
        raise NotImplemented()
        
    def fit(self, features, classes):
        # fit your model to the provided features
        raise NotImplemented()
        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        raise NotImplemented()