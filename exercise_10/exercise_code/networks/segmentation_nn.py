"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as function

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        resnet = models.resnet50(pretrained=True)

        self.model = nn.Sequential(*list(resnet.children())[:6])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tconv1 = ConvLayer(512, 256)
        self.bn1 = nn.BatchNorm2d(256,eps=1e-05, momentum=0.01, affine=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tconv2 = ConvLayer(256, 128)
        self.bn2 = nn.BatchNorm2d(128,eps=1e-05, momentum=0.01, affine=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tconv3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_classes,eps=1e-05, momentum=0.01, affine=True)

    
     


        pass 
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        for param in self.model.parameters():
            param.requires_grad = True  
        x = self.model(x)
        x = function.relu(self.bn1(self.tconv1(self.up1(x))))
        x = function.relu(self.bn2(self.tconv2(self.up2(x))))
        x = function.relu(self.bn3(self.tconv3(self.up3(x))))
        

      
        pass
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")