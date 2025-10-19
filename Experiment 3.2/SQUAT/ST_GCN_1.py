import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ST_GCN(nn.Module):

    """"
    This class is the full model that brings everything together to recognize the excises from skeleton keypoint data over time.
    To take joint keypoints over time and predict what action is being performed by:
    1. Learning how joints relate to each other.
    2. Learning how joints move over time.
    3. Using multiple layers to extract higher-level motion patterns.
    4. Making a final prediction using a fully connected layer.
    """
    def __init__(self, num_classes, channels=3, num_joints=25): #runs when you create a new ST_GCN_1 model.
        super(ST_GCN, self).__init__()

        #Graph config
        self.graph = {'layout': 'coco',  #model uses the COCO dataset's joint format
                    'strategy': 'spatial'} #Uses spatial relationships between joints to build connections
        
        self.matrix = self.matrix(num_joints) #creates the adjacency matrix

        #stacked ST-GCN blocks, which perform both spatial and temporal convolution
        self.st_gcn_blocks = nn.ModuleList([
            ST_GCN_Block(channels, 64, self.matrix, residual=False),
            ST_GCN_Block(64, 64, self.matrix),
            ST_GCN_Block(64, 64, self.matrix),
            ST_GCN_Block(64, 128, self.matrix, stride=1),
            ST_GCN_Block(128, 128, self.matrix),
            ST_GCN_Block(128, 128, self.matrix),  
            ST_GCN_Block(128, 256, self.matrix, stride=1),
            ST_GCN_Block(256, 256, self.matrix),
            ST_GCN_Block(256, 256, self.matrix),  
        ])
        
        self.fc = nn.Linear(256, num_classes)
    #___________________________________________________________________________________________________________________________
    #This function creates an adjacency matrix for a human skeleton graph.
    def matrix(self, num_joints):
        edges = [
            # Torso
            (0, 1), (1, 2), (1, 5), (1, 8),
            # Right arm
            (2, 3), (3, 4),
            # Left arm  
            (5, 6), (6, 7),
            # Hip to legs
            (8, 9), (8, 12),
            # Right leg
            (9, 10), (10, 11),
            # Left leg
            (12, 13), (13, 14),
            # Head
            (0, 15), (0, 16), (15, 17), (16, 18),
            ####£
            (11, 22), (11, 24), (22, 23),
            (14, 19), (14, 21), (19, 20)
        ]

        matrix = torch.zeros((num_joints, num_joints)) #Labels the matrix with 0s
        for i, j in edges:
            matrix[i, j] = matrix[j, i] = 1  #Puts 1s where the is a connection  between nodes
        return matrix
    
    #___________________________________________________________________________________________________________________________
    #The forward function in this class defines how input data moves through the model from raw pose data to final class prediction

    
    def forward(self, x):
        #input shape of x is [N, C, T, V]
        for gcn in self.st_gcn_blocks:
            x = gcn(x) #Moves x through each ST-GCN block, one after the other
        
        # Global average pooling
        x = nn.functional.avg_pool2d(x, (x.size()[2], x.size()[3]))
        x = x.view(x.size(0), -1)  #Combines channels and joints into a single vector per sample
        return self.fc(x)
    

###################################################################################################################################
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix, residual=True, stride=1, temporal_kernel_size=3):
        super(ST_GCN_Block, self).__init__()
        
        # Spatial graph convolution
        self.gcn = Spatial_GCN(in_channels, out_channels, adjacency_matrix)
        
        # Temporal convolution - FIXED
        # Calculate proper padding to maintain temporal dimension
        padding = (temporal_kernel_size // 2, 0)  # This ensures output has same temporal length
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), 
                     padding=padding, stride=(stride, 1)),  # Fixed kernel size and padding
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.3)  
        )
        
        # Residual connection
        self.residual = residual
        if residual:
            # Only downsample if dimensions don't match
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ) if (in_channels != out_channels) or (stride != 1) else nn.Identity()
        else:
            self.downsample = None
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        x = self.gcn(x)
        x = self.tcn(x)
        
        if self.residual:
            residual = self.downsample(residual)
            x += residual
            
        return self.relu(x)

#####################################################################################################################################
class Spatial_GCN(nn.Module):
    """"
    This class applies a graph convolution over the spatial structure (i.e., joints) of the pose in a single frame.
    It uses the adjacency matrix 'matrix' to guide how information flows between joints.
    """
    def __init__(self, in_channels, out_channels, matrix):
        """
        in_channels: Number of input features per joint e.g. (x, y, z, visibility)
        out_channels: Number of output features after applying the graph conv
        matrix: Adjacency matrix for 17 joints. It's a 17x17 matrix
        self.conv: A 1x1 convolution that just changes the number of channels per joint, without mixing across time or joints
        """
        super(Spatial_GCN, self).__init__()
        self.matrix = matrix
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    #___________________________________________________________________________________________________________________________
    def forward(self, x):
        x = self.conv(x) 
        n, c, t, v = x.size()
        x = torch.einsum('nctv,vw->nctw', (x, self.matrix.to(x.device)))  #Multiplies the joint features with the adjacency matrix.
        return x