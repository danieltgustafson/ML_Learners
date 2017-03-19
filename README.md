# GT_ML_HW
Projects for my CS MS program

This was the second project for ML for Trading course.  We were tasked with building our own Random Tree code.  A random tree is an interesting implementation of a Decision Tree.

Traditional Decision Trees (done in another assignment in this project) actually use a metric to determine what feature and value would be best to split the data on.  Usually this would be something like Information Gain / Entropy.

For a Random Tree, rather than actually optimize the split, we instead simply take a random feature and choose random values to split on.  Surprisingly, little predictive power is lost - and computation is significantly quicker/easier.

We next had to implement a 'Bag Learner' - similar to a Random Forest - we create multiple random trees using different cuts of the same input data (with replacement) and take the response value from the multiple trees.

