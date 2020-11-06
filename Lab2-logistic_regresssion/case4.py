import numpy as np
import logistic_regression as lg

train_data = np.array([[170, 80, 1],
						[90, 15, 0],
						[130, 30, 0],
						[165, 55, 1],
						[150, 45, 1],
						[120, 40, 0],
						[110, 35, 0],
						[180, 70, 1],
						[175, 65, 1],
						[160, 60, 1]],
						dtype = np.float16)

test_data = np.array([[170, 60],
					 [85, 15],
					 [145, 45]],
					 dtype = np.float16)

model = lg.LogisticRegression(lr=0.00045, datas=train_data, epoch=350000, err = 0.01)
init_w = np.copy(model.w)
model.train()
lg.draw(model=model, init_w=init_w, train_data=train_data, test_data=test_data)


