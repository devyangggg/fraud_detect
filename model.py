# ## LSTM Model

import torch
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
    

x = torch.randn(32, 100, 18)  
print(model(x).shape)          

loss_func = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

steps = 100

for epoch in range(steps):
    optimiser.zero_grad()
    output = model(x)
    output = output.squeeze()
    loss = loss_func(output,y)
    loss.backward()
    optimiser.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")