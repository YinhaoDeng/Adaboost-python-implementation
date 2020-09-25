import numpy as np


dataset = np.genfromtxt('wdbc_data.csv', delimiter=',', dtype=str)  # load data from csv file using genfromtxt
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
# print(train_data, train_data.shape)

# distribution = np.full((1, x.shape[1]), 1/x.shape[0])  # initialise distribution
# print(distribution.shape)

# sort train data  TODO: Do we need to make a copy of train data or not?
# sorted_data_x, sorted_data_y = sorted_data[:, 0], sorted_data[:, 1:]  # seperate label and features
# for idx in range(30):  # range should be 30


class DecisionStumps:
    error_rate = -1
    split_criteria_idx = -1
    split_num = -1

    def __init__(self):
        pass

    def sort_train_data_by_column(self, split_criteria_idx, train_data):
        return train_data[train_data[:, split_criteria_idx+1].argsort()]

    def split_sorted_train_data(self, split_criteria_idx, split_num, sorted_train_data):
        self.split_criteria_idx = split_criteria_idx
        self.split_num = split_num

        left_sorted_train_data, right_sorted_train_data = [], []
        for data in sorted_train_data:
            if data[split_criteria_idx + 1] < split_num:
                left_sorted_train_data.append(data)
                print("<")
            elif data[split_criteria_idx + 1] >= split_num:
                right_sorted_train_data.append(data)
                print(">")
        return left_sorted_train_data, right_sorted_train_data

    def calculate_error_rate(self, original_x):
        error_rate = 0  # TODO: calculate error rate
        self.error_rate = error_rate

    def calculate_alpha_weight(self):
        alpha = 0.5 * np.log((1-self.error_rate)/self.error_rate)
        return alpha


a = DecisionStumps()
sorted_train_data = a.sort_train_data_by_column(split_criteria_idx=0,
                                          train_data=train_data)
left, right = a.split_sorted_train_data(split_criteria_idx=0, split_num=10, sorted_train_data=sorted_train_data)
print(left)
print('----'*10)
print(right)



