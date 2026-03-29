import torch
import torch.nn as nn
from sklearn.metrics import classification_report  
from data_gen import train_data, test_data
from model import model


loss_func = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

steps = 100

for epoch in range(steps):
    total_train_loss = 0
    model.train()
    for x_batch, y_batch in train_data:
        optimiser.zero_grad()
        output = model(x_batch).squeeze()
        loss = loss_func(output,y_batch)
        total_train_loss += loss
        loss.backward()
        optimiser.step()

    total_train_loss = total_train_loss/len(train_data)

    #------------------
    
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_data:
            output = model(x_batch).squeeze()
            loss = loss_func(output,y_batch)
            total_test_loss += loss

        total_test_loss = total_test_loss/len(test_data)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {total_train_loss}, Test Loss: {total_test_loss}")
    


model.eval()
with torch.no_grad():
    all_output = []
    all_y = []
    for x_batch, y_batch in test_data:
        output = (torch.sigmoid(model(x_batch)) > 0.5).int()
        all_output.append(output)
        all_y.append(y_batch)

    all_output = torch.cat(all_output)  
    all_y = torch.cat(all_y)  
    report = classification_report(all_y.numpy(),all_output.squeeze().numpy())

print(report)

torch.save(model.state_dict(), "main_model.pt")