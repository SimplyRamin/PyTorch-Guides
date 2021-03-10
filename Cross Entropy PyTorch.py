import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

# 3 samples -> n_samples = 3
Y = torch.tensor([2, 0, 1])
# this tensor means that first row of y_hat should have highest value in 2 index or 3rd column
# second row of y_hat should have highest value in 0 index or first column
# third row of y_hat should have highest value in 1 index or second columns

# y_hat tensor size should be -> n_samples x n_classes 
# for this example we will use -> 3 x 3
Y_hat_good = torch.tensor([[.1, 1.0, 2.1],
                           [2.0, 1.0, .1],
                           [.1, 3.0, .1]
                        ])
Y_hat_bad = torch.tensor([[2.1, 1.0, .1],
                          [.1, 1.0, 2.1],
                          [.1, 3.0, .1]
                        ])

l1 = loss(Y_hat_good, Y)
l2 = loss(Y_hat_bad, Y)

print(f'Good Prediction loss: {l1}')
print(f'Bad Prediction loss: {l2}')

_, prediction1 = torch.max(Y_hat_good, 1)
_, prediction2 = torch.max(Y_hat_bad, 1)

print(f'Predicted Y of Good Prediction: {prediction1}')
print(f'Predicted Y of bad Prediction: {prediction2}')