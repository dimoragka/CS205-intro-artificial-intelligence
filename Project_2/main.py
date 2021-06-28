import numpy as np
import math
from numba import jit
import time


def euclidean_distance(x, y):
    distance = 0
    for i, j in zip(x, y):
        distance += (i - j) ** 2
    return math.sqrt(distance)
    #return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def nn_loo_accuracy(X,y):
    """
    Leave-one out nearest neighbour implementation.
    """
    # number of correct votes
    correct_votes = 0
    # number of incorrect votes
    incorrect_votes = 0
    num_instances = X.shape[0]
    for i in range(num_instances):
        min_distance = pow(10,20)
        nearest_neighbor = -1
        for j in range(num_instances):
            if i == j: continue
            distance = euclidean_distance(X[i], X[j])
            if distance < min_distance:
                nearest_neighbor = j
                min_distance = distance
        if (y[i] == y[nearest_neighbor]):
            correct_votes += 1
        else:
            incorrect_votes += 1
    return (correct_votes / num_instances) * 100


@jit(nopython=True)
def euclidean_distance_numba(x, y):
    """
    Euclidean distance calculation with Numba accelaration (http://numba.pydata.org).
    """
    distance = 0
    for i, j in zip(x, y):
        distance += (i - j) ** 2
    return math.sqrt(distance)


@jit(nopython=True)
def nn_loo_accuracy_numba(X,y):
    """
    Leave-one out nearest neighbour implementation with Numba accelaration (http://numba.pydata.org).
    """

    # number of correct votes
    correct_votes = 0
    # number of incorrect votes
    incorrect_votes = 0
    num_instances = X.shape[0]
    for i in range(num_instances):
        min_distance = pow(10,20)
        nearest_neighbor = -1
        for j in range(num_instances):
            if i == j: continue
            distance = euclidean_distance_numba(X[i], X[j])
            if distance < min_distance:
                nearest_neighbor = j
                min_distance = distance
        if (y[i] == y[nearest_neighbor]):
            correct_votes += 1
        else:
            incorrect_votes += 1
    return (correct_votes / num_instances) * 100


def forward_selection(X, y, knn_func):
    """
    Forward-search feature selection using Leave-One-Out cross-validator
    with classifier implementing the k-nearest neighbors vote.

    """
    num_features = X.shape[1]
    best_accuracy = 0
    # initially empty set of best features
    best_features = set()
    print("\nBegin Forward-Selection Search")
    for i in range(0, num_features):
        #print('\n-- On Level #' + str(i+1) + ' of the search tree:')
        feature_to_add_on_this_level = -1
        local_best_accuracy = 0
        for j in range(0, num_features):
            if j not in best_features:
                test_features = best_features.copy()
                test_features.add(j)
                X_test = X[:,np.asarray(list(test_features), dtype=int)]
                accuracy = knn_func(X_test, y)
                # add 1 to print in 1 to num_features range
                print_test_features = set([i+1 for i in test_features])
                print("  Using feature(s) {} accuracy is {}%".format(print_test_features, accuracy))
                if accuracy > local_best_accuracy:
                    local_best_accuracy = accuracy
                    feature_to_add_on_this_level = j
        if(local_best_accuracy > best_accuracy):
            print("\n  Level #{} decision: I added feature {} to best features set an accuracy increased from {} to {}".
                  format(i, feature_to_add_on_this_level+1, best_accuracy, local_best_accuracy))
            best_accuracy = local_best_accuracy
            best_features.add(feature_to_add_on_this_level)
        else:
            print('\n  Level #{} decision: Accuracy will decreased if a new feature will be added to best features'.format(i))
            break
    print('\nForward-Selection search completed...')
    # add 1 to print in 1 to num_features range
    print_best_features = set([i + 1 for i in best_features])
    print('\n  Best feature(s) are ' + ''.join(str(print_best_features)) + ' with an accuracy of ' + str(best_accuracy) + '%\n')


def backward_elimination(X, y, knn_func):
    """
    Backward-elimination feature selection using Leave-One-Out cross-validator
    with classifier implementing the k-nearest neighbors vote.

    """
    num_features = X.shape[1]
    # starting accuracy is the accuracy on all features
    best_accuracy = nn_loo_accuracy(X, y)
    # initially all features are in the best
    best_features = set([i for i in range(num_features)])
    print("\nBegin Backward-Elimination Search")
    for i in range(0, num_features):
        print('\n-- On Level #' + str(i + 1) + ' of the search tree:')
        feature_to_remove_on_this_level = -1
        local_best_accuracy = 0
        for j in range(0, num_features):
            if j in best_features:
                test_features = best_features.copy()
                test_features.remove(j)
                X_test = X[:,np.asarray(list(test_features), dtype=int)]
                accuracy = knn_func(X_test, y)
                # add 1 to print in 1 to num_features range
                print_test_features = set([i+1 for i in test_features])
                print("  Using feature(s) {} accuracy is {}%".format(print_test_features, accuracy))
                if accuracy > local_best_accuracy:
                    local_best_accuracy = accuracy
                    feature_to_remove_on_this_level = j
        if(local_best_accuracy > best_accuracy):
            print("\n  Level #{} decision: I removed feature {} from best features set and accuracy increased from {} to {}".
                  format(i, feature_to_remove_on_this_level+1, best_accuracy, local_best_accuracy))
            best_accuracy = local_best_accuracy
            best_features.remove(feature_to_remove_on_this_level)
        else:
            print('\n  Level #{} decision: Accuracy will decreased if a new feature will be removed from the best features'.format(i))
            break
    print('\nBackward-Elimination search completed...')
    # add 1 to print in 1 to num_features range
    print_best_features = set([i + 1 for i in best_features])
    print('\n  Best feature(s) are ' + ''.join(str(print_best_features)) + ' with an accuracy of ' + str(best_accuracy) + '%\n')


if __name__ == "__main__":
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Welcome to Paraskevi\'s Program for Feature Selection with Nearest Neighbor.')
    parser.add_argument('filepath', type=argparse.FileType('r'), help='path of file containing the data')  # https://stackoverflow.com/questions/18862836/how-to-open-file-using-argparse
    parser.add_argument('-s', '--search_method', required=True, type=int, help='1 Forward selection, 2 for Backward Elimination,  3 for Paraskevi\'s Optimized Forward selection and 4 for Paraskevi\'s Optimized Backward selection')
    args = parser.parse_args()

    # read data from file
    # https://cmdlinetips.com/2011/08/three-ways-to-read-a-text-file-line-by-line-in-python/
    data = []
    with args.filepath as file:
        line = file.readline()  # use readline() to read the first line
        while line:
            elements = line.rstrip().split()  # split line into its elements
            row = []
            row.append(int(float(elements[0])))
            row.extend([float(e) for e in elements[1:]])
            data.append(row)  # first element is the label an integer...
            line = file.readline()

    # convert list to numpy array
    data = np.asarray(data)
    # labels index 0 of data
    y = data[:,0].astype(int)
    # features are from index 1 to the last index of data
    X = data[:,1:]
    # no longer necessary
    data = []

    if args.search_method == 1:
        start = time.perf_counter()
        forward_selection(X, y, nn_loo_accuracy)
        end = time.perf_counter()
        elapsed = end - start
        print("Forward selection took {:.12f} seconds".format(elapsed))
    elif args.search_method == 2:
        start = time.perf_counter()
        backward_elimination(X, y,nn_loo_accuracy)
        end = time.perf_counter()
        elapsed = end - start
        print("Backward selection took {:.12f} seconds".format(elapsed))
    elif args.search_method == 3:
        start = time.perf_counter()
        forward_selection(X, y, nn_loo_accuracy_numba)
        end = time.perf_counter()
        elapsed = end - start
        print("Paraskevi's Optimized Forward selection took {:.12f} seconds".format(elapsed))
    elif args.search_method == 4:
        start = time.perf_counter()
        backward_elimination(X, y, nn_loo_accuracy_numba)
        end = time.perf_counter()
        elapsed = end - start
        print("Paraskevi's Optimized Backward selection took {:.12f} seconds".format(elapsed))
    else:
        parser.print_help()
