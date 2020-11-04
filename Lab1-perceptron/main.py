import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
	def __init__(self, datas, lr, epoch):
		self.lr = lr
		self.datas = datas
		self.epoch = epoch
		self.flag = True
		self.w = np.random.uniform(0, 0.1, 3)

	def sign(self, x):
		return 1 if x > 0 else -1;

	def update(self, delta_w):
		# w += learning_rate*delta_w
		self.w += (self.lr*delta_w)

	def train(self):
		now_epoch = 0
		while True:
			now_epoch += 1
			self.flag = True
			for data in self.datas:
				# [w0, w1, w2] [1, x1, x2]
				h = np.dot(self.w, np.array([1,data[0],data[1]]))
				# if y != h(x)
				if data[2] != self.sign(h):
					self.flag = False
					self.update(data[2]*np.array([1,data[0],data[1]]))

			# all correct or reach limited epoch
			if self.flag == True or now_epoch >= self.epoch:	
				print("w0: {}, w1: {}, w2: {}".format(self.w[0], self.w[1], self.w[2]))
				break;

	def test(self, data):
		prediction = np.dot(self.w, np.array([1, data[0], data[1]]))
		return self.sign(prediction)


if __name__ == '__main__':
	train_data = np.loadtxt("train.txt", dtype=np.float16, delimiter=',')
	model = Perceptron(datas=train_data, lr=0.001, epoch=1000)
	model.train()	# train

	# draw
	w = model.w
	x = np.linspace(-25, 30)
	y = -((w[0]+w[1]*x)/w[2])
	plt.figure()
	plt.plot(x, y, linewidth=1)	# final weight
	for data in train_data:
		if data[2] == 1:
			plt.scatter(data[0], data[1], s=1, c='r', marker='o')
		else:
			plt.scatter(data[0], data[1], s=1, c='b', marker='o')

	# test
	test_data = np.loadtxt("test.txt", dtype=np.float16, delimiter=',')
	for data in test_data:
		result = model.test(data)
		if result == 1:
			plt.scatter(data[0], data[1], s=10, c='r', marker='v')
		else:
			plt.scatter(data[0], data[1], s=10, c='b', marker='v')
		print(data, result)

	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.title('Training and Test Data')
	train_po = plt.scatter(0, 0, s=1, c='r', marker='o')
	train_ne = plt.scatter(0, 0, s=1, c='b', marker='o')
	test_po = plt.scatter(0, 0, s=10, c='r', marker='v')
	test_ne = plt.scatter(0, 0, s=10, c='b', marker='v')
	plt.legend([train_po, train_ne, test_po, test_ne], ["+1", "-1", "+1", "-1"])
	plt.show()



	
