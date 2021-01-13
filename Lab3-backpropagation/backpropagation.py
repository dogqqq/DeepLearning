import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Backpropagation(object):
	
	def __init__(self, lr, train_datas, valid_datas, epoch, train_labels, valid_labels):
		super(Backpropagation, self).__init__()
		self.lr = lr
		self.w1 = np.random.uniform(-0.1, 0.1, (10, 784))
		self.w2 = np.random.uniform(-0.1, 0.1, (3, 10))
		self.b1 = np.random.uniform(-0.1, 0.1, (10, 1))
		self.b2 = np.random.uniform(-0.1, 0.1, (3, 1))
		self.train_datas = train_datas
		self.valid_datas = valid_datas
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.epoch = epoch

	def sigmoid(self, n):
		return 1/(1+np.exp(-n))

	def train(self):

		for i in range(self.epoch):
			
			cnt = 0
			for data, label in zip(self.train_datas, self.train_labels):

				data = data.reshape(784, 1)
				label = label.reshape(3,1)

				# forward
				n1 = np.dot(self.w1, data) + self.b1
				a1 = self.sigmoid(n1)
				
				n2 = np.dot(self.w2, a1) + self.b2
				a2 = self.sigmoid(n2)

				# train_acc
				if np.argmax(a2) == np.argmax(label):
					cnt += 1

				# backward
				d2 = a2 - label
				d1 = (a1*(1-a1)) * np.dot(self.w2.transpose(), d2)

				# update
				self.w2 = self.w2 - self.lr*np.dot(d2, a1.transpose())
				self.w1 = self.w1 - self.lr*np.dot(d1, data.transpose())
				self.b2 = self.b2 - self.lr*d2 
				self.b1 = self.b1 - self.lr*d1

			print("epoch: ", i, "train accuracy: ", cnt/len(self.train_datas))

			# valid
			cnt = 0
			for data, label in zip(self.valid_datas, self.valid_labels):

				data = data.reshape(784, 1)

				# forward
				n1 = np.dot(self.w1, data) + self.b1;
				a1 = self.sigmoid(n1)

				n2 = np.dot(self.w2, a1) + self.b2;
				a2 = self.sigmoid(n2)

				# train_acc
				if np.argmax(a2) == label:
					cnt += 1			

			print("valid accuracy: ", cnt/len(self.valid_datas))

	def test(self, test_data, save_to):
		with open(save_to, "w") as f:
			for data in test_data:
				data = data.reshape(784, 1)

				n1 = np.dot(self.w1, data) + self.b1;
				a1 = self.sigmoid(n1)

				n2 = np.dot(self.w2, a1) + self.b2;
				a2 = self.sigmoid(n2)

				f.write("{}\n".format(np.argmax(a2)))


# read data
img = pd.read_table('./train_img.txt', header=None, dtype=np.float64, sep=',').to_numpy()
img_label = pd.read_table('./train_label.txt', header=None, sep=',').to_numpy()
test_img = pd.read_table('./test_img.txt', header=None, sep=',').to_numpy()

data = np.hstack([img, img_label]) # image and label concate
data = np.random.permutation(data) 	# shuffle
img = data[ : , 0 : 784]	# image
img_label = data[ : , 784 : ]	# label

train_datas = img[ : 6400, : ]
valid_datas = img[ 6400 : , : ]	# split data, 8000 * 0.8

train_labels = OneHotEncoder().fit_transform(img_label[ : 6400, : ]).toarray()
valid_labels = img_label[ 6400 : , : ]

# normalize picture from (0, 255) to (-1, 1)
train_datas = (train_datas-127.5)/127.5
valid_datas = (valid_datas-127.5)/127.5

model = Backpropagation(lr=0.01, train_datas=train_datas, valid_datas=valid_datas, epoch=150, train_labels=train_labels, valid_labels=valid_labels)
model.train()
print("layer: (784, 10, 3), learning rate: 0.01, epoch: 150")

# read test data
test_img = pd.read_table('./test_img.txt', header=None, sep=',').to_numpy()
test_img = (test_img-127.5)/127.5

model.test(test_data=test_img, save_to="test.txt")
