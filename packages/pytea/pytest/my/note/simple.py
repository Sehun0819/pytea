import torch
import torch.nn as nn
import json

def weights_to_list(state_dict):
    state_dict_ = {}
    for k, v in state_dict.items():
        state_dict_[k] = v.tolist()
    return state_dict_

def extract_weights(net, file_path=None):
    weights = net.state_dict()
    json_string = json.dumps(weights_to_list(weights), sort_keys=True, indent=4)
    if file_path == None:
        print(json_string)
    else:
        with open(file_path, 'w') as file:
            file.write(json_string)

model = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)

out = model(torch.rand(10,1,20,20))
extract_weights(model, 'weights.json')
"""
t1 = torch.rand(2,3,4)
t2 = torch.rand(4,6)
t3 = torch.matmul(t1, t2)
t4 = torch.rand(3,6)
t5 = t3 + t4
h = t5.toh()

t6 = t1.flatten()
h6 = t6.toh()
"""

"""
t1 = torch.rand(5, 30)
fc = nn.Linear(30, 10)
t2 = fc(t1) + 3
"""