"""cnns for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection cnn"""
    def __init__(self, hparams):
        """
        Initialize your cnn from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        self.batch_size = hparams['batch_size']
        num_filters = hparams['num_filters']
        kernel_size = hparams['kernel_size']
        stride = hparams['stride']
        padding = hparams['padding']
        dropout = hparams['dropout_rate']
        self.cnn = nn.ModuleList()
        in_ch = 1
        for i in range(1,5):
            self.cnn.append(nn.Conv2d(in_channels=in_ch, out_channels=i*num_filters, kernel_size=kernel_size, stride=stride, padding=padding))
            self.cnn.append(nn.ReLU())
            self.cnn.append(nn.MaxPool2d(2, 2))
            self.cnn.append(nn.Dropout(dropout+i*0.1))
            in_ch = num_filters*i
            nn.init.kaiming_normal_(self.cnn[-4].weight)
        self.cnn = nn.Sequential(*self.cnn)
        
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(4608, 256))
        self.mlp.append(nn.Linear(256, hparams['num_keypoints']*2))
        self.mlp = nn.Sequential(*self.mlp)
        nn.init.xavier_normal_(self.mlp[0].weight)
        nn.init.xavier_normal_(self.mlp[1].weight)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your cnn                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.cnn(x)
        x = x.reshape(x.shape[0], 1, -1)
        x = self.mlp(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy cnn always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
