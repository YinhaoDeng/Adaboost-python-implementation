import numpy as np


class DecisionStumps:
    error_rate = -1
    split_criteria_idx = -1
    split_num = -1
    reverse = False
    alpha_weight = -1

    def __init__(self):
        pass

    def show_object_info(self):
        print("error rate: {}, split_criteria_idx: {}, split number: {}, reversed or not: {}, alpha weight {}".
              format(self.error_rate, self.split_criteria_idx, self.split_num, self.reverse, self.alpha_weight))


####################################################################
def sort_train_data_by_column(split_criteria_idx, train_data):
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
    print("<", a, "    >", b)
    return left_sorted_train_data, right_sorted_train_data


def split_sorted_train_data_and_calculate_error_rate(split_criteria_idx, split_num, sorted_train_data, reverse_case):
    left_sorted_train_data, right_sorted_train_data = split_sorted_train_data(split_criteria_idx, split_num, sorted_train_data)
    # total_sample_weight = np.sum(sorted_train_data, axis=1)[-2]
    # print("total sample weight: ", total_sample_weight)
    wrong_weight_normal = 0
    wrong_weight_reverse = 0

    for data in left_sorted_train_data:
        if data[-1] == -1:
            # wrong_num_reverse += 1
            wrong_weight_reverse += data[-2]  # add weight of wrong samples
        elif data[-1] == 1:
            # wrong_num_normal += 1
            wrong_weight_normal += data[-2]  # add weight of wrong samples

    for data in right_sorted_train_data:
        if data[-1] == -1:
            # wrong_num_normal += 1
            wrong_weight_normal += data[-2]  # add weight of wrong samples
        elif data[-1] == 1:
            # wrong_num_reverse += 1
            wrong_weight_reverse += data[-2]  # add weight of wrong samples

    if not reverse_case:
        # error_rate = wrong_num_normal/(len(left_sorted_train_data)+len(right_sorted_train_data))
        # error_rate = wrong_weight_normal/total_sample_weight
        error_rate = wrong_weight_normal  # /1  total weight sum = 1
    elif reverse_case:
        # error_rate = wrong_num_reverse/(len(left_sorted_train_data)+len(right_sorted_train_data))
        error_rate = wrong_weight_reverse  # /1
    return error_rate


def calculate_alpha_weight(error_rate):
    alpha_weight = 0.5 * np.log((1-error_rate)/error_rate)
    return alpha_weight


def update_sample_weight():
    pass


###################################################################################################################
dataset = np.genfromtxt('wdbc_data.csv', delimiter=',', dtype=str)  # load data from csv file using genfromtxt
dataset = np.delete(dataset, 0, 1)  # delete first column because it's useless ????? TODO: do we need this step?

# convert label from string (M/B) into int (1/-1)
for data in dataset:
    if data[0] == 'M':
        data[0] = 1
    elif data[0] == 'B':
        data[0] = -1

train_data = dataset[:300].astype(np.float)  # string format to float
initial_weight = 1/train_data.shape[0]
sample_weight_column = np.full((train_data.shape[0], 1), initial_weight)  # add sample weight column to training set
train_data = np.hstack((train_data, sample_weight_column))
train_data = np.roll(train_data, -1)  # move sample label to the last for computation convenience
print(train_data)

test_data = dataset[300:].astype(np.float)  # string format to float
test_data = np.roll(test_data, -1)  # move sample label to the last for computation convenience


# select the split with minimum error rate:
T = 1  # Global value
feature_nums = train_data.shape[1]-1  # should be 30 features
week_classifier_list = [DecisionStumps() for i in range(T)]  # initialise DecisionStumps class object list

for t in range(T):  # have T iterations overall
    # select minimum error split criteria and split num
    min_error_rate = 1
    for feature_idx in range(feature_nums):
        print("feature index: ", feature_idx)
        sorted_train_data = sort_train_data_by_column(split_criteria_idx=feature_idx, train_data=train_data)

        for idx, data in enumerate(train_data[:-1]):  # iterate from 0 index to second last index
            print(idx)
            split_num = (train_data[idx][feature_idx] + train_data[idx+1][feature_idx])/2

            # calculate error rate in both cases
            normal_case_error_rate = split_sorted_train_data_and_calculate_error_rate(split_criteria_idx=feature_idx,
                                                                                split_num=split_num,
                                                                                sorted_train_data=sorted_train_data,
                                                                                reverse_case=False)
            reversed_case_error_rate = split_sorted_train_data_and_calculate_error_rate(split_criteria_idx=feature_idx,
                                                                               split_num=split_num,
                                                                               sorted_train_data=sorted_train_data,
                                                                               reverse_case=True)

            print("normal case error: ", normal_case_error_rate)
            print("reversed case error: ", reversed_case_error_rate)

            # update if new best week classifier is found
            if normal_case_error_rate < min_error_rate or reversed_case_error_rate < min_error_rate:
                if normal_case_error_rate < min_error_rate:
                    print("new min_error_rate (1)")
                    week_classifier_list[t].error_rate = normal_case_error_rate
                    week_classifier_list[t].reverse = False
                    min_error_rate = normal_case_error_rate
                elif reversed_case_error_rate < min_error_rate:
                    print("new min_error_rate (2)")
                    week_classifier_list[t].error_rate = reversed_case_error_rate
                    week_classifier_list[t].reverse = True
                    min_error_rate = reversed_case_error_rate
                week_classifier_list[t].split_num = split_num
                week_classifier_list[t].split_criteria_idx = feature_idx
                week_classifier_list[t].alpha_weight = calculate_alpha_weight(min_error_rate)
            print("min error rate: ", min_error_rate)

            # TODO： update sample weights



            # quit()
        # quit()

print("#"*100)
week_classifier_list[0].show_object_info()



