import torch
import csv
import ast
from torch import nn
import numpy as np
from tqdm import trange

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            
            nn.Linear(50, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 1000),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
        


with open('training_data.csv', newline='') as f:
    reader = csv.reader(f)
    training_data = [(row[0], int(row[1]), np.asarray(ast.literal_eval(row[2])).reshape(25,2)) for row in reader]
    
# remove all data with label -1

training_data = [d for d in training_data if d[1] != -1]
    
    
# with open('test_data.csv', newline='') as f:
#     reader = csv.reader(f)
#     test_data = [(row[0], int(row[1]), np.asarray(ast.literal_eval(row[2])).reshape(25,3)) for row in reader]
    

# test_data = [d for d in test_data if d[1] != -1]

#train the nn

model = NeuralNetwork()

# convert training data to tensors

training_data = [(name, label, torch.tensor(points.flatten(), dtype=torch.float32)) for name, label, points in training_data]
# test_data = [(name, label, torch.tensor(points.flatten(), dtype=torch.float32)) for name, label, points in test_data]

#only take 0.1 of the training data

training_data = training_data[:int(len(training_data)*0.1)]

#split this into training and validation data

test_data = training_data[int(len(training_data)*0.8):]
training_data = training_data[:int(len(training_data)*0.8)]


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


NUM_EPOCHS = 1000
BATCH_SIZE = 32

trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

tq = trange(NUM_EPOCHS, desc='Loss: ', leave=True)
running_loss = 0.0
for epoch in tq:
    tq.set_description('Loss: {:.4f}'.format(running_loss/len(trainloader)))
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        name, labels, inputs = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        
            
print("Finished training")

