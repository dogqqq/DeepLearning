import numpy as np
import logistic_regression as lg

train_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype = np.float16)
test_data = np.array([])

model = lg.LogisticRegression(lr=0.01, datas=train_data, epoch=10000, err = 0.01)
init_w = np.copy(model.w)
model.train()
lg.draw(model=model, init_w=init_w, train_data=train_data, test_data=test_data)


