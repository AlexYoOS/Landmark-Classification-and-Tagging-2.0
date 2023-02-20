import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
# define the CNN architecture
class MyModel(nn.Module):
    ## TODO: choose an architecture, and complete the class
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()
        
        ## Define layers of a CNN
        # (3x224x224)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        #(6x112x112)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        # (16x56x56)
        self.conv2_bn = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # (32x28x28)
        self.conv3_bn = nn.BatchNorm2d(32)
        # pool
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected layers
         
        self.fc1 = nn.Linear(32*28*28, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, num_classes)
       

        # drop-out
        self.dropout = nn.Dropout(dropout)
        
        
    
    def forward(self, x):
        ## Define forward behavior
        
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))
        
        #flatten batch_size
        
        x = x.view(-1, 32*28*28)
        
        x = self.dropout(x)
        x = self.fc1(x)
        
        x = F.relu(self.fc1_bn(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        
        x = self.dropout(x)
        x = self.fc3(x)
        
        
        
        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
