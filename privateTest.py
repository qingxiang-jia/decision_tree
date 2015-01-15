from learning import *

# Part I example
# learner = DecisionTreeLearner()
# learner.train(zoo)
# learner.prune(zoo, 5)
# learner.dt.display()
# print '# of nodes =', learner.dt.count_nodes()
# learner.dt.outputDecisionTree("./output.py")

# Part II Test
test_set = DataSet(name='../data/test_data', attrnames='X1 X2 Y', target='Y')
learner = DecisionTreeLearner()
learner.train(test_set)
learner.prune(test_set, 5)
learner.dt.display()