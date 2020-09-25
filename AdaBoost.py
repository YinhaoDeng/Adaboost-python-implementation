import numpy as np


dataset = np.genfromtxt('data/wdbc_data.csv', delimiter=',', dtype=str)  # load data from csv file using genfromtxt
dataset = np.delete(dataset, 0, 1)  # delete first column because it's useless ????? TODO: do we need this step?

# convert label from string type into int type
for data in dataset:
    if data[0] == 'M':
        data[0] = 1
    elif data[0] == 'B':
        data[0] = -1


# split data into train and test datasets
train_data = dataset[:300].astype(np.float)  # string to float
test_data = dataset[300:].astype(np.float)  # string to float
# print(test_data, test_data.shape)

# distribution = np.full((1, x.shape[1]), 1/x.shape[0])  # initialise distribution
# print(distribution.shape)

# sort train data  TODO: Do we need to make a copy of train data or not?
# sorted_data = train_data(np.argsort(train_data[:, split_criteria_idx + 1]))  # because the label hasn't been seperated
# sorted_data_x, sorted_data_y = sorted_data[:, 0], sorted_data[:, 1:]  # seperate label and features
# for idx in range(len(sorted_data.shape[0]) - 1):  # range should be 30


class DecisionStumps:
    error_rate = -1
    split_criteria_idx = -1
    split_num = -1

    def __init__(self):
        pass

    # def sort_train_data_based_on_criteria(self, split_criteria_idx, split_num):
    #     self.split_criteria_idx = split_criteria_idx
    #     self.split_num = split_num




    def calculate_error_rate(self, original_x):
        error_rate = 0  # TODO: calculate error rate
        self.error_rate = error_rate

    def calculate_alpha_weight(self):
        alpha = 0.5 * np.log((1-self.error_rate)/self.error_rate)
        return alpha


# a = DecisionStumps()
# print(a.split_num)


