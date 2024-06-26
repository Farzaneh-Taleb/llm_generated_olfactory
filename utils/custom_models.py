from torch import nn
import torch.nn.functional as F
from deepchem.models.torch_models.torch_model import TorchModel
# from openpom.utils.loss import CustomMultiLabelLoss,CustomMultiLabelBCEWitLogitsLoss
# from openpom.utils.optimizer import get_optimizer
from deepchem.models.losses import Loss, L2Loss

import torch
class LinearModelMultiLabelDeepchem(TorchModel):
    def __init__(self,model,class_imbalance_ratio,device_name,loss_aggr_type,mode,optimizer_name,learning_rate,batch_size):

        if mode == 'regression':
            loss: Loss = L2Loss()
            # output_types: List = ['embedding','prediction']
            output_types: List = ['prediction']
        else:
            loss = CustomMultiLabelBCEWitLogitsLoss(
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=device_name)

# When predict() is called, only the first output (the probabilities) will be returned. But during training, it is the second output (the logits) that will be passed to the loss function.
            output_types = ['prediction', 'loss', 'embedding']
        
        optimizer: Optimizer = get_optimizer(optimizer_name)
        optimizer.learning_rate = learning_rate
        if device_name is not None:
            device: Optional[torch.device] = torch.device(device_name)
        else:
            device = None
        super(LinearModelMultiLabelDeepchem, self).__init__(model,loss,output_types=output_types,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           device=device_name)
        
        
class NonLinearModel(nn.Module):
    def __init__(self,n_tasks):
        super(NonLinearModel, self).__init__()
        self.n_tasks=n_tasks
        self.fc1 = nn.Linear(768, 512) # 12 is the number of features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.n_tasks)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = F.relu(self.fc3(x))
        out = self.fc4(embeddings)
    
        if self.n_tasks == 1:
            logits = out.view(-1, self.n_classes)
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        proba = torch.sigmoid(logits)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        
 
        return proba, logits, embeddings


class LinearModel(nn.Module):
    def __init__(self,n_tasks,n_dim):
        super(LinearModel, self).__init__()
        self.n_tasks=n_tasks
        self.n_dim = n_dim
        self.fc3 = nn.Linear(n_dim, 256)
        self.fc4 = nn.Linear(256, self.n_tasks)
    
    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)
        embeddings = self.fc3(x)
        out = self.fc4(embeddings)
    
        if self.n_tasks == 1:
            logits = out.view(-1, self.n_classes)
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        proba = torch.sigmoid(logits)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        
 
        return logits, logits, logits


class LinearModelAlva(nn.Module):
    def __init__(self,n_tasks):
        super(LinearModelAlva, self).__init__()
        self.n_tasks=n_tasks
        self.fc1 = nn.Linear(21, 512) # 12 is the number of features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.n_tasks)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = F.relu(self.fc3(x))
        out = self.fc4(embeddings)
    
        if self.n_tasks == 1:
            logits = out.view(-1, self.n_classes)
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        proba = torch.sigmoid(logits)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        
 
        return proba, logits, embeddings



class LinearModelSmall(nn.Module):
    def __init__(self,n_tasks):
        super(LinearModelSmall, self).__init__()
        self.n_tasks=n_tasks
        self.fc3 = nn.Linear(21, 16)
        self.fc4 = nn.Linear(16, self.n_tasks)
    
    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        embeddings = F.relu(self.fc3(x))
        out = self.fc4(embeddings)
    
        if self.n_tasks == 1:
            logits = out.view(-1, self.n_classes)
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        proba = torch.sigmoid(logits)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        
 
        return proba, logits, embeddings


class LinearModelRegression(nn.Module):
    def __init__(self,n_tasks):
        super(LinearModelRegression, self).__init__()
        self.n_tasks=n_tasks
        self.fc1 = nn.Linear(32, 16) # 12 is the number of features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, self.n_tasks)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = F.relu(self.fc3(x))
        out = self.fc4(embeddings)
    
        if self.n_tasks == 1:
            proba = out.view(-1, self.n_classes)
        else:
            proba = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        # return embeddings,proba
        # print("out",out)
        # print("outsize",out.size())
        return out


class LinearModelRegression2(nn.Module):
    def __init__(self,n_tasks):
        super(LinearModelRegression2, self).__init__()
        self.n_tasks=n_tasks
        # self.fc1 = nn.Linear(768, 512) # 12 is the number of features
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(10, self.n_tasks)
    
    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # embeddings = F.relu(self.fc3(x))
        out = self.fc4(x)
    
        if self.n_tasks == 1:
            proba = out.view(-1, self.n_classes)
        else:
            proba = out.view(-1, self.n_tasks, self.n_classes)
        # proba = torch.sigmoid(logits)  # (batch, n_tasks, classes)
        if self.n_classes == 1:
            proba = proba.squeeze(-1)  # (batch, n_tasks)
        
        # return embeddings,proba
        # print("out",out)
        # print("outsize",out.size())
        return out