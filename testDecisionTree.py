# test a classification algorithm
from learning import *

car_train = DataSet(name='../data/car/car_train', attrnames='buying maint doors persons lug_boot safety values',
                    target='values')
car_cv = DataSet(name='../data/car/car_cv', attrnames='buying maint doors persons lug_boot safety values',
                 target='values')
car_test = DataSet(name='../data/car/car_test', attrnames='buying maint doors persons lug_boot safety values',
                   target='values')

train_set = car_train
test_set = car_test
cv_set = car_cv


def run_test(learner, maxDeviation,
             learning_curve_interval, test_trials, random_shuffle,
             train_set, cv_set, test_set):
    learner.train(train_set)
    if maxDeviation > 0:
        learner.prune(cv_set, maxDeviation)
    print "Macro F1 score on test data: ", "%.5f" % test(learner, test_set)
    res = learningcurve(learner, maxDeviation, train_set, cv_set, test_set,
                        learning_curve_interval, test_trials, random_shuffle=random_shuffle)
    print "train dataset size: ", [x[0] for x in res]
    print "average node count on test dataset: ", ["%.2f" % x[1][0] for x in res]
    print "average F1 score on test dataset: ", ["%.5f" % x[1][1] for x in res]


learner = DecisionTreeLearner()
interval = 100

# test
# print "Before Pruning"
# print "========================="
# run_test(learner, 0, interval, 1, False, train_set, cv_set, test_set)

# test
# print "After Pruning"
# print "========================="
# run_test(learner, 8, interval, 1, False, train_set, cv_set, test_set)

# Massive test - added by qj2125

# A helper function to iterate in fraction
def f_range(start, end=None, inc=None):
    "A range function that accepts float increments"
    if end == None:
        end = start + 0.0
        start = 0.0
    if inc == None:
        inc = 1.0
    lst = []
    while 1:
        next = start + len(lst) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        lst.append(next)
    return lst

print "Before Pruning"
print "========================="
run_test(learner, 0, interval, 1, False, train_set, cv_set, test_set)

print 'Find best MaxDev'
for maxDev in f_range(0, 10, 0.1):
    print 'maxDev =', maxDev
    print "After Pruning"
    print "========================="
    run_test(learner, maxDev, interval, 1, False, train_set, cv_set, cv_set)

print 'For plotting graphs'
for maxDev in f_range(0, 10, 0.1):
    print 'maxDev =', maxDev
    print "After Pruning"
    print "========================="
    run_test(learner, maxDev, interval, 1, False, train_set, cv_set, test_set)

print 'Final test'
print "After Pruning"
print "========================="
run_test(learner, 3.6, interval, 1, False, train_set, cv_set, test_set)
