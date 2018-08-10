

RTLearner.py implements the 'random tree' method that simplifies a standard decision tree by ignoring entropy / information gain
and simply selects a random feature for splitting at each level.

BagLearner.py is an add-on for RTLearner.py which utilizes the Random Trees script and implements bagging.

decision_trees.py implements decision_trees in the more classical fashion by utilizing entropy/IG to select the best feature at each level.
This script also contains a random_forest class.
