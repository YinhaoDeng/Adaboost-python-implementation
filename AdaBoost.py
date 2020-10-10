import numpy as np
import matplotlib.pyplot as plt



class DecisionStumps:
    error_rate = -1
    split_criteria_idx = -1
    split_num = -1
    reverse = False
    alpha_weight = -1
    accuracy = -1

    left_data, right_data = [], []


    def __init__(self):
        pass

    def show_object_info(self):
        print("error rate: {:.6f}, accuracy {:.2}  |  [split_criteria_idx: {}, split number: {:.4}]  | reversed or not: {}, alpha weight {:.5}".
              format(self.error_rate, self.accuracy, self.split_criteria_idx, self.split_num, self.reverse, self.alpha_weight))


####################################################################
def sort_train_data_by_column(split_criteria_idx, train_data):
    print(split_criteria_idx, len(train_data))
    return train_data[train_data[:, split_criteria_idx].argsort()]


def split_sorted_train_data(split_criteria_idx, split_num, sorted_train_data):
    left_sorted_train_data, right_sorted_train_data = [], []
    a, b = 0, 0
    for data in sorted_train_data:
        if data[split_criteria_idx] < split_num:
            left_sorted_train_data.append(data)
            a += 1
        elif data[split_criteria_idx] >= split_num:
            right_sorted_train_data.append(data)
            b += 1
    print("<", a, "    >=", b)
    return left_sorted_train_data, right_sorted_train_data


def calculate_error_rate(left_sorted_train_data, right_sorted_train_data, reverse_case):
    wrong_weight_normal, wrong_weight_reverse = 0, 0
    correct_num_normal, correct_num_reverse = 0, 0

    if reverse_case is False:  # normal case  {-1, 1}
        for data in left_sorted_train_data:  # {-1}
            if data[-1] == -1:  # correctly classified
                correct_num_normal += 1
            elif data[-1] == 1:  # wrongly classified
                wrong_weight_normal += data[-2]
        for data in right_sorted_train_data:  # {1}
            if data[-1] == -1:  # correctly classified
                wrong_weight_normal += data[-2]
            elif data[-1] == 1:  # wrongly classified
                correct_num_normal += 1
        error_rate = wrong_weight_normal
        accuracy = correct_num_normal / sorted_train_data.shape[0]

    elif reverse_case:  # reverse case {1, -1}
        for data in left_sorted_train_data:  # {1}
            if data[-1] == -1:  # wrongly classified
                wrong_weight_reverse += data[-2]
            elif data[-1] == 1:
                correct_num_reverse += 1
        for data in right_sorted_train_data:  # {-1}
            if data[-1] == -1:
                correct_num_reverse += 1
            elif data[-1] == 1:
                wrong_weight_reverse += data[-2]
        error_rate = wrong_weight_reverse
        accuracy = correct_num_reverse / sorted_train_data.shape[0]
    return error_rate, accuracy


def calculate_alpha_weight(error_rate):
    alpha_weight = 0.5 * np.log((1-error_rate)/error_rate)
    return alpha_weight


def update_sample_weight_and_update_train_data(left_sorted_train_data, right_sorted_train_data, reverse_case, alpha_weight):
    if reverse_case is False:
        for data in left_sorted_train_data:
            data[-2] = data[-2] * np.exp(-alpha_weight*data[-1] * (-1))
        for data in right_sorted_train_data:
            data[-2] = data[-2] * np.exp(-alpha_weight * data[-1] * (1))

    elif reverse_case:
        for data in left_sorted_train_data:
            data[-2] = data[-2] * np.exp(-alpha_weight*data[-1] * (1))
        for data in right_sorted_train_data:
            data[-2] = data[-2] * np.exp(-alpha_weight*data[-1] * (-1))

    # calculate total sum of sample weights, Z_t, the normalization factor

    if left_sorted_train_data:
        sample_weight_left = np.sum(left_sorted_train_data, axis=0)[-2]  # sum along the column
    else:
        sample_weight_left = 0

    if right_sorted_train_data:
        sample_weight_right = np.sum(right_sorted_train_data, axis=0)[-2]  # sum along the column
    else:
        sample_weight_right = 0
    total_sample_weight = sample_weight_left + sample_weight_right

    # normalise data
    for data in left_sorted_train_data:
        data[-2] /= total_sample_weight
    for data in right_sorted_train_data:
        data[-2] /= total_sample_weight

    new_train_data = np.append(left_sorted_train_data, right_sorted_train_data, axis=0)
    return new_train_data


def predict(test_data, week_clf_list, t):
    result = []

    for data in test_data:
        negative_say, positive_say = 0, 0

        for clf in week_clf_list[:t+1]:
            if data[clf.split_criteria_idx] < clf.split_num:
                if clf.reverse is False:  # {-1, 1}
                    negative_say += clf.alpha_weight
                if clf.reverse:  # {1, -1}
                    positive_say += clf.alpha_weight
            elif data[clf.split_criteria_idx] >= clf.split_num:
                if clf.reverse is False:  # {-1, 1}
                    positive_say += clf.alpha_weight
                if clf.reverse:  # {1, -1}
                    negative_say += clf.alpha_weight

        if negative_say > positive_say:
            result.append(-1)
        elif negative_say <= positive_say:
            result.append(1)

    return result


def cal_test_acc(test_data, predict_result):
    correct_num = 0
    total_test_num = test_data.shape[0]
    for i in range(total_test_num):
        if test_data[i][-1] == predict_result[i]:
            correct_num += 1
    test_acc = correct_num / total_test_num
    return test_acc


###################################################################################################################
dataset = np.genfromtxt('wdbc_data.csv', delimiter=',', dtype=str)  # load data from csv file using genfromtxt
dataset = np.delete(dataset, 0, 1)  # delete first column because it's useless

# convert label from string (M/B) into int (1/-1)
for data in dataset:
    if data[0] == 'M':
        data[0] = 1
    elif data[0] == 'B':
        data[0] = -1

train_data = dataset[:300].astype(np.float)  # string format to float
train_data_sklearn = train_data[:, 1:]
initial_weight = 1/train_data.shape[0]
sample_weight_column = np.full((train_data.shape[0], 1), initial_weight)  # add sample weight column to training set
train_data = np.hstack((train_data, sample_weight_column))
label_column = train_data[:, 0].reshape(train_data.shape[0], 1)
train_label_sklearn = train_data[:, 0]

train_data = np.hstack((train_data[:, 1:], label_column))

#######################################################################################################
test_data = dataset[300:].astype(np.float)  # string format to float
# test_data = np.roll(test_data, -1)  # move sample label to the last for computation convenience
test_label_column = test_data[:, 0].reshape(test_data.shape[0], 1)
test_data_sklearn = test_data[:, 1:]
test_label_sklearn = test_data[:, 0]
test_data = np.hstack((test_data[:, 1:], test_label_column))

# select the split with minimum error rate:
feature_nums = train_data.shape[1]-2  # should be 30 features [Why -2?  Because we have weight column & label column]

week_classifier_list = [DecisionStumps() for i in range(feature_nums)]  # initialise DecisionStumps class object list

hello = []
steps = 30

for feature_idx in range(feature_nums):
    min_error_rate = 1
    sorted_train_data = sort_train_data_by_column(split_criteria_idx=feature_idx, train_data=train_data)
    # min_num = sorted_train_data[0][feature_idx]
    # max_num = sorted_train_data[-1][feature_idx]
    # print(min_num, max_num)

    # for i in range(-1, sorted_train_data.shape[0]+1):  # iterate from 0 index to 298
    for i in range(sorted_train_data.shape[0]-1):  # iterate from 0 index to 298
        # split_num = min_num + i * (max_num - min_num) / steps
        split_num = (sorted_train_data[i][feature_idx] + sorted_train_data[i+1][feature_idx])/2
        # print("feature_idx", feature_idx, "data[i]: ", sorted_train_data[i][feature_idx], "data[i+1]", sorted_train_data[i+1][feature_idx], "split value: ", split_num)

        left_sorted_train_data, right_sorted_train_data = split_sorted_train_data(split_criteria_idx=feature_idx,
                                                                                  split_num=split_num,
                                                                                  sorted_train_data=sorted_train_data)

        # calculate error rate in both cases
        normal_case_error_rate, normal_acc = calculate_error_rate(left_sorted_train_data=left_sorted_train_data,
                                                                  right_sorted_train_data=right_sorted_train_data,
                                                                  reverse_case=False)
        reversed_case_error_rate, reversed_acc = calculate_error_rate(left_sorted_train_data=left_sorted_train_data,
                                                                      right_sorted_train_data=right_sorted_train_data,
                                                                      reverse_case=True)

        print("normal case error: ", normal_case_error_rate, "noraml acc: ", normal_acc)
        print("reversed case error: ", reversed_case_error_rate, "reversed acc: ", reversed_acc)

        # update if new best week classifier is found
        if normal_case_error_rate < min_error_rate or reversed_case_error_rate < min_error_rate:
            if normal_case_error_rate < min_error_rate:
                # print("new min_error_rate (1)")
                week_classifier_list[feature_idx].error_rate = normal_case_error_rate
                week_classifier_list[feature_idx].reverse = False
                week_classifier_list[feature_idx].accuracy = normal_acc
                min_error_rate = normal_case_error_rate
                flag = False
            elif reversed_case_error_rate < min_error_rate:
                # print("new min_error_rate (2)")
                week_classifier_list[feature_idx].error_rate = reversed_case_error_rate
                week_classifier_list[feature_idx].reverse = True
                week_classifier_list[feature_idx].accuracy = reversed_acc
                min_error_rate = reversed_case_error_rate
                flag = True

            week_classifier_list[feature_idx].split_num = split_num
            week_classifier_list[feature_idx].split_criteria_idx = feature_idx
            week_classifier_list[feature_idx].alpha_weight = calculate_alpha_weight(min_error_rate)
            week_classifier_list[feature_idx].left_data = left_sorted_train_data
            week_classifier_list[feature_idx].right_data = right_sorted_train_data

        print("---current min error rate: ", min_error_rate, "reverse ? ", flag, '\n')

    result = predict(test_data, week_classifier_list, feature_idx)
    test_acc = cal_test_acc(test_data, result)
    hello.append(test_acc)

    # update train_data
    train_data = update_sample_weight_and_update_train_data(left_sorted_train_data=week_classifier_list[feature_idx].left_data,
                                                             right_sorted_train_data=week_classifier_list[feature_idx].right_data,
                                                             reverse_case=week_classifier_list[feature_idx].reverse,
                                                             alpha_weight=week_classifier_list[feature_idx].alpha_weight)
    print("weight updated! train_data updated!")


print("#"*100)
for idx, clf in enumerate(week_classifier_list):
    print(idx, clf.show_object_info())

for idx, data in enumerate(hello):
    print('t=', idx, ', accuracy= ', data)

# plot accuracy against iterations using matplot
plt.figure(figsize=(5,4))
plt.title('AdaBoost Accuracy against Iterations')
plt.xlabel('iteration number')
plt.ylabel('accuracy')
t = range(len(hello))

plt.plot(t, hello)
plt.show()




#####################################################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
print("################   sklearn Adaboost  #################")
clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30, random_state=0,
    algorithm="SAMME.R", learning_rate=0.5)
clf.fit(train_data_sklearn, train_label_sklearn)
# clf.predict(test_data_sklearn)
sklearn_acc = clf.score(test_data_sklearn, test_label_sklearn)
print("sklearn accuracy: {:.3}%".format(sklearn_acc*100))

