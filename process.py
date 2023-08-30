import torch
from torch import nn
import numpy as np
import scipy.io
from datetime import datetime

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

annotations = scipy.io.loadmat('./mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat', struct_as_record=False)

release = annotations['RELEASE']

must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]

def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret

def print_dataset_obj(obj, depth = 0, maxIterInArray = 20):
    prefix = "  "*depth
    if type(obj) == dict:
        for key in obj.keys():
            print("{}{}".format(prefix, key))
            print_dataset_obj(obj[key], depth + 1)
    elif type(obj) == list:
        for i, value in enumerate(obj):
            if i >= maxIterInArray:
                break
            print("{}{}".format(prefix, i))
            print_dataset_obj(value, depth + 1)
    else:
        print("{}{}".format(prefix, obj))

# Convert to dict
dataset_obj = generate_dataset_obj(release)

data = dataset_obj['annolist']

training_data = []

#generate training dataset

for i in range(len(data)):
    d = dataset_obj['annolist'][i]
    name = d['image']['name']
    points = [(0,0)]*15

    try:
        for p in d['annorect']:
            if 'annopoints' in p:
                for point in p['annopoints']['point']:
                    if point['is_visible'] == 0:
                        continue
                    points[point['id']] = (point['x'], point['y'])
        label = dataset_obj['act'][i]['act_id']
        if dataset_obj['img_train'][i] == 0:
            continue
        # convert points to tensor
        points = points.astype(np.int16)
        training_data.append((points, label))
        
        
        
    except:
        print("Error in {}".format(name))
    
    
# generate test dataset

test_data = []


for i in range(len(data)):
    d = dataset_obj['annolist'][i]
    name = d['image']['name']
    points = [(0,0)]*15

    try:
        for p in d['annorect']:
            if 'annopoints' in p:
                for point in p['annopoints']['point']:
                    if point['is_visible'] == 0:
                        continue
                    points[point['id']] = (point['x'], point['y'])
        label = dataset_obj['act'][i]['act_id']
        if dataset_obj['img_train'][i] == 1:
            continue
        
        points = points.astype(np.int16)
        test_data.append((points, label))
        
        
        
    except:
        print("Error in {}".format(name))

# print(training_data)

# initalize model

#three leaky relu layers and a hyperbolic tangent layer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(15, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 15),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device=device)

# set up loss function and optimizer

loss_fn = nn.MSELoss(reduction='sum')

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# train model

def train_one_epoch(epoch, model, loss_fn, optimizer, device, tb_writer):
    running_loss = 0.0
    last_loss = 0.0
    
    for (X, y) in enumerate(training_data):
        batch+=1
        X = X.to(device)
        y = y.to(device)
        
        # Compute prediction error
        
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch % 10 == 9:
            last_loss = running_loss / 10
            print(f"Epoch: {epoch}, Batch: {batch}, Average Loss: {last_loss}")
            tb_x = epoch * len(training_data) + batch + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0
            
            
    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, model, loss_fn, optimizer, device, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(test_data):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
        