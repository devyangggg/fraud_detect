import torch.nn as nn

class Anomaly_Detect(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=18, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=64, out_features=1)

    def forward(self,x):
        out, _ = self.lstm(x)
        res = self.linear(out[:,-1,:])

        return res
        

model = Anomaly_Detect()
  