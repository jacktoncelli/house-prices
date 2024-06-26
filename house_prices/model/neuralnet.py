import torch.nn as nn

class NeuralNetHousePrice(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output, dropout_prob = 0.5):
        super(NeuralNetHousePrice, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1) 
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, hidden3)
        self.l4 = nn.Linear(hidden3, output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l4(out)

        return out
