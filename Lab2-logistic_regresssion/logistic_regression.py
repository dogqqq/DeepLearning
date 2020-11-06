import math
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
	def __init__(self, lr, epoch, datas, err):
		self.lr = lr
		self.datas = datas
		self.epoch = epoch
		self.err_boundary = err
		self.w = np.random.uniform(0, 0.1, 3)

	def sigmoid(self, n):
		return 1/(1+np.exp(-n))

	def update(self, delta_w):
		self.w += (self.lr*delta_w)

	def train(self):
		N = self.datas.shape[0]
		now_epoch = 0
		err = 0.0
		for now_epoch in range(self.epoch):
			err = 0.0

			for data in self.datas:
				# [w0, w1, w2] * [1, x1, x2]
				n = np.dot(self.w, np.array([1,data[0],data[1]]))
				a = self.sigmoid(n)
				# cross_error = -(y*ln(a) + (1-y)*ln(1-a))
				err += (-(data[2]*np.log(a)+(1-data[2])*np.log(1-a)))
				self.update((data[2]-a)*np.array([1,data[0],data[1]]))

			# print(err/N)
			if err/N < self.err_boundary:
				print("Error: {} < {} (small enough)".format(err/N, self.err_boundary))
				break

		print("w0: {}, w1: {}, w2: {}".format(self.w[0], self.w[1], self.w[2]))	
		print("Stop Epoch: {}, Total Epoch: {}".format(now_epoch+1, self.epoch))
		if now_epoch+1 == self.epoch:
			print("Error: {}".format(err/N))
	

	def predict(self, test_data):
		probability = self.sigmoid(np.dot(self.w, np.array([1, test_data[0], test_data[1]])))
		if probability >= 0.5:
			return 1
		else:
			return 0


def draw(init_w, train_data, test_data, model):
	_max = float("-inf")
	_min = float("inf")

	for i in train_data:
		_max = max(i[0], i[1], _max)
		_min = min(i[0], i[1], _min)

	if test_data.shape[0] != 0:
		for i in test_data:
			_max = max(i[0], i[1], _max)
			_min = min(i[0], i[1], _min)

	x = np.linspace(_min-5, _max+5)
	init_y = -((init_w[0]+init_w[1]*x)/init_w[2])
	final_y = -((model.w[0]+model.w[1]*x)/model.w[2])
	plt.figure()
	init, = plt.plot(x, init_y, linewidth=1, linestyle='--', color='r')
	final, = plt.plot(x, final_y, linewidth=1)

	train_po = train_ne = test_po = test_ne = 0

	# training_data
	for data in train_data:
		if data[2] == 1:
			train_po = plt.scatter(data[0], data[1], s=1, c='r', marker='o')
		else:
			train_ne = plt.scatter(data[0], data[1], s=1, c='b', marker='o')

	# test_data
	for data in test_data:
		result = model.predict(data)
		if result == 1:
			test_po = plt.scatter(data[0], data[1], s=10, c='r', marker='v')
		else:
			test_ne = plt.scatter(data[0], data[1], s=10, c='b', marker='v')
		print("[x1, x2]: ", data, "prediction: ", result)

	plt.ylabel('x2')
	plt.xlabel('x1')

	tot_cat_list = [train_po, train_ne, test_po, test_ne]
	tot_name_list = ["Init", "Train", "1 (Train)", "0 (Train)", "1 (Test)", "0 (Test)"]
	cat_list = [init, final]
	name_list = ["Init", "Train"]

	# in case no correspond label dot
	cnt = 2
	for i in tot_cat_list:
		if i != 0:
			cat_list.append(i)
			name_list.append(tot_name_list[cnt])
		cnt += 1

	if test_data.shape[0] != 0:
		plt.title('Training and Test Data')
	else:
		plt.title('Training Data')
	plt.legend(cat_list, name_list)
	plt.show()
