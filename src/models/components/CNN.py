import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=14, kernel_size = 3, hidden_dim=32,out_dim=1, dropout=0.5):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_channels
        self.dropout=dropout
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=kernel_size, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Flatten(),
            nn.Linear(8, self.hidden_dim))
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2,  self.out_dim))
        self.model = nn.Sequential(self.encoder, self.regressor)
    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src= src.view(src.size(0),self.input_dim,-1)
        # features = self.encoder(src)
        # predictions = self.regressor(features)
        predictions = self.model(src)
        return predictions
#cnn_model=CNN_RUL(14,30,0.5)
#cnn_model.encoder

