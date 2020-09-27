import numpy as np

class DecisionStumps:
    error_rate = -1
    split_criteria_idx = -1
    split_num = -1
    reverse = False

    def __init__(self):
        pass

    def sort_train_data_by_column(self, split_criteria_idx, train_data):
        return train_data[train_data[:, split_criteria_idx].argsort()]

    def split_sorted_train_data(self, split_criteria_idx, split_num, sorted_train_data):
        self.split_criteria_idx = split_criteria_idx
        self.split_num = split_num

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

    def calculate_error_rate(self, left_sorted_train_data, right_sorted_train_data, reverse_case): # TODO: 2cases: (-1, 1), (1, -1)
        wrong_num_normal = 0  # case (-1, 1)
        wrong_num_reverse = 0  # case (1, -1)

        for data in left_sorted_train_data:
            if data[-1] == -1:
                wrong_num_reverse += 1
            elif data[-1] == 1:
                wrong_num_normal += 1

        for data in right_sorted_train_data:
            if data[-1] == -1:
                wrong_num_normal += 1
            elif data[-1] == 1:
                wrong_num_reverse += 1

        error_rate_normal = wrong_num_normal/(len(left_sorted_train_data)+len(right_sorted_train_data))
        error_rate_reverse = wrong_num_reverse/(len(left_sorted_train_data)+len(right_sorted_train_data))
        self.error_rate = error_rate_normal if reverse_case is False else error_rate_reverse

    def calculate_alpha_weight(self):
        alpha = 0.5 * np.log((1-self.error_rate)/self.error_rate)
        return alpha


###################################################################################################################

dataset = np.genfromtxt('wdbc_data.csv', delimiter=',', dtype=str)  # load data from csv file using genfromtxt
dataset = np.delete(dataset, 0, 1)  # delete first column because it's useless ????? TODO: do we need this step?

# convert label from string type into int type
for data in dataset:
    if data[0] == 'M':
        data[0] = 1
    elif data[0] == 'B':
        data[0] = -1

dataset = np.roll(dataset, -1)  # [features + label] instead of [label + features] for convinience
print(dataset)

# split data into train and test datasets
train_data = dataset[:300].astype(np.float)  # string to float
test_data = dataset[300:].astype(np.float)  # string to float
# print(train_data, train_data.shape)

# distribution = np.full((1, x.shape[1]), 1/x.shape[0])  # initialise distribution
# print(distribution.shape)

# select the split with minimum error rate:
T = 1  # Global value
feature_nums = train_data.shape[1]-1  # should be 30 features
# print("feature nums: ", feature_nums)
# quit()
week_classifier_list = [DecisionStumps() for i in range(T)] # initialise DecisionStumps class object list

for t in range(T):  # have T iterations overall
    # select minimum error split criteria and split num
    for feature_idx in range(feature_nums):
        print("feature index: ", feature_idx)
        week_classifier_list[t].sort_train_data_by_column(split_criteria_idx=feature_idx, train_data=train_data)
        for idx, data in enumerate(train_data):
            if idx < train_data.shape[0]-1:  # if current line is not the last line : < 300
                print(idx, data)
                split_num = (train_data[idx][feature_idx] + train_data[idx+1][feature_idx])/2
                print(split_num)
            # if idx == 5:
            #     quit()
                week_classifier_list[t].split_sorted_train_data(split_criteria_idx=feature_idx, )

quit()
#
#
# a = DecisionStumps()
# sorted_train_data = a.sort_train_data_by_column(split_criteria_idx=0,
#                                                 train_data=train_data)
# left, right = a.split_sorted_train_data(split_criteria_idx=0, split_num=10, sorted_train_data=sorted_train_data)
# # print(left)
# print('----'*10)
# # print(right)
# a.calculate_error_rate(left, right, reverse_case=True)
# print("error rate: ", a.error_rate)
# a.calculate_error_rate(left, right, reverse_case=False)
# print("error rate reverse: ", a.error_rate)
# print("alpha weight ", a.calculate_alpha_weight())


