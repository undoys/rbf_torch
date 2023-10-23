import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RBF(nn.Module):
    def __init__(self, x, y, step_num):
        super(RBF, self).__init__()
        self.step_num = step_num
        self.learning_rate, self.hidden_size = self.calculate_hyperparameters(x, y)
        
    def calculate_hyperparameters(self, x, y):
        n_samples, n_features = np.shape(x)
        n_outputs = np.shape(y)[1]
        learning_rate = 0.1 / np.sqrt(n_samples * n_features)
        hidden_size = int(np.sqrt(n_samples * n_outputs))
        return learning_rate, hidden_size

    def kernel(self, x):
        x1 = x.repeat(self.hidden_size, 1)
        x2 = x1.reshape(-1, self.hidden_size, self.feature_x)
        # c = self.c.repeat(np.shape(x)[0], 1, 1)
        c= self.c.repeat(np.shape(x)[0], self.hidden_size, 1)
        s = self.s.repeat(np.shape(x)[0], self.hidden_size, 1)
        dist = torch.sum((x2 - c) ** 2 /(2* s ** 2), 2)
        return torch.exp(-dist)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        z = self.kernel(x)
        yf = torch.matmul(z, self.w) + self.b
        return yf.detach().numpy()
    
    def train(self, x, y):
        self.feature_x = np.shape(x)[1]
        self.feature_y = np.shape(y)[1]
        x_ = torch.tensor(x, dtype=torch.float32)
        y_ = torch.tensor(y, dtype=torch.float32)
        # self.c = nn.Parameter(torch.randn(self.hidden_size, self.feature_x))
        self.c = nn.Parameter(torch.randn(self.feature_x))
        self.s = nn.Parameter(torch.randn(self.feature_x))
        self.w = nn.Parameter(torch.randn(self.hidden_size, self.feature_y))
        # self.w = nn.Parameter(torch.randn(self.feature_y))
        self.b = nn.Parameter(torch.zeros(self.feature_y))
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # loss_fn = nn.MSELoss()
        with open('loss.txt', 'w') as f:
            for epoch in range(self.step_num+1):
                optimizer.zero_grad()
                z = self.kernel(x_)
                yf = torch.matmul(z, self.w) + self.b
                # loss = torch.mean(torch.square(yf-y_))
                # loss = loss_fn(yf, y_)
                loss = F.mse_loss(yf, y_)
                loss.backward()
                optimizer.step()
                f.write(str(loss.item())+'\n')
                
                # Adjust the learning rate and hidden size based on the loss
                if epoch % 500 == 0 and epoch != 0 and loss.item() > 0.01:
                    self.learning_rate /= 2
                    self.hidden_size += 1
                    self.w = nn.Parameter(torch.randn(self.hidden_size, self.feature_y))
                    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            f.close()

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        z = self.kernel(x)
        yf = torch.matmul(z, self.w) + self.b
        return yf.detach().numpy()
