import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pickle


def init(key, module, weights=None):
    # key : layer name
    # value: weights at certain layer
    if weights is None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    # module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    # module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])
    if isinstance(module, ConvBlock):
        if (key + "_1").encode() in weights and (key + "_0").encode() in weights:
            module.conv.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
            module.conv.weight.data = torch.from_numpy(weights[(key + "_0").encode()])
    else:
        if (key + "_1").encode() in weights and (key + "_0").encode() in weights:
            module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
            module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])
    
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        # self.b1 = nn.Sequential(
        #     ConvBlock(in_channels, n1x1, kernel_size=1, stride=1, padding=0)
        # )

        # # 1x1 -> 3x3 conv branch
        # self.b2 = nn.Sequential(
        #     ConvBlock(in_channels, n3x3red, kernel_size=1, stride=1, padding=0),
        #     ConvBlock(n3x3red, n3x3, kernel_size=3, stride=1, padding=1)
        # )

        # # 1x1 -> 5x5 conv branch
        # self.b3 = nn.Sequential(
        #     ConvBlock(in_channels, n5x5red, kernel_size=1, stride=1, padding=0),
        #     ConvBlock(n5x5red, n5x5, kernel_size=5, stride=1, padding=2)
        # )

        # # 3x3 pool -> 1x1 conv branch
        # self.b4 = nn.Sequential(
        #     nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
        #     ConvBlock(in_channels, pool_planes, kernel_size=1, stride=1, padding=0)
        # )
        self.b1 = nn.Sequential(init(f'{key}/1x1', ConvBlock(in_channels, n1x1, 1, 1, 0), weights))
        self.b2 = nn.Sequential(
            init(f'{key}/3x3_reduce', ConvBlock(in_channels, n3x3red, 1, 1, 0), weights),
            init(f'{key}/3x3', ConvBlock(n3x3red, n3x3, 3, 1, 1), weights)
        )
        self.b3 = nn.Sequential(
            init(f'{key}/5x5_reduce', ConvBlock(in_channels, n5x5red, 1, 1, 0), weights),
            init(f'{key}/5x5', ConvBlock(n5x5red, n5x5, 5, 1, 2), weights)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            init(f'{key}/pool_proj', ConvBlock(in_channels, pool_planes, 1, 1, 0), weights)
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels):
        super(InceptionAux, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)  # 528 -> 128
        self.act = nn.ReLU()  
        self.fc1 = nn.Linear(2048, 1024)  # 2048 -> 1024
        self.dropout = nn.Dropout(0.7)  # 70% dropout
        self.fc2 = nn.Linear(1024, 3)  # 1024 -> 3, for xyz position
        self.fc3 = nn.Linear(1024, 4)  # 1024 -> 4, for wpqr orientation
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.act(x)
    
        x = torch.flatten(x, 1) # [batch_size, channels, height, width] -> [batch_size, features]
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x1 = self.fc2(x) # for xyz position
        x2 = self.fc3(x) # for wpqr orientation
        
        return x1, x2


class LossHeader(nn.Module):
    def __init__(self, key=None, weights=None):
        super(LossHeader, self).__init__()
        
        # TODO: Define loss headers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 2048)  # Extra layer as per requirements
        self.dropout = nn.Dropout(0.4)
        self.fc_xyz = nn.Linear(2048, 3)
        self.fc_wpqr = nn.Linear(2048, 4)

        # if weights is not None and key is not None:
        #     # Initialize weights for each layer with the specified key
        #     self.fc = init(f'{key}_fc', self.fc, weights)
        #     self.fc_xyz = init(f'{key}_fc_xyz', self.fc_xyz, weights)
        #     self.fc_wpqr = init(f'{key}_fc_wpqr', self.fc_wpqr, weights)

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [batch_size, channels, height, width] -> [batch_size, features]
        x = self.fc(x)
        x = self.dropout(x)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            # print(weights.keys())
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        # self.pre_layers = nn.Sequential(
        #     # Example for defining a layer and initializing it with pretrained weights
        #     init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),

        # )
        
        # Example for InceptionBlock initialization
        # self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)

        # self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        # self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        # self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        # self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.conv1 = init('conv1/7x7_s2', ConvBlock(3, 64, 7, 2, 3), weights)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = init('conv2/3x3_reduce', ConvBlock(64, 64, 1, 1, 0), weights)
        self.conv3 = init('conv2/3x3', ConvBlock(64, 192, 3, 1, 1), weights)
        self.pool3 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)

        # self.inception3A = InceptionBlock(in_channels=192,
        #                                    n1x1=64,
        #                                    n3x3red=96,
        #                                    n3x3=128,
        #                                    n5x5red=16,
        #                                    n5x5=32,
        #                                    pool_planes=32)
        # self.inception3B = InceptionBlock(in_channels=256,
        #                                    n1x1=128,
        #                                    n3x3red=128,
        #                                    n3x3=192,
        #                                    n5x5red=32,
        #                                    n5x5=96,
        #                                    pool_planes=64)
        self.inception3A = InceptionBlock(192, 64, 96, 128, 16, 32, 32, 'inception_3a', weights)
        self.inception3B = InceptionBlock(256, 128, 128, 192, 32, 96, 64, 'inception_3b', weights)

        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        # self.inception4A = InceptionBlock(in_channels=480,
        #                                    n1x1=192,
        #                                    n3x3red=96,
        #                                    n3x3=208,
        #                                    n5x5red=16,
        #                                    n5x5=48,
        #                                    pool_planes=64)
        # self.inception4B = InceptionBlock(in_channels=512,
        #                                    n1x1=160,
        #                                    n3x3red=112,
        #                                    n3x3=224,
        #                                    n5x5red=24,
        #                                    n5x5=64,
        #                                    pool_planes=64)
        # self.inception4C = InceptionBlock(in_channels=512,
        #                                    n1x1=128,
        #                                    n3x3red=128,
        #                                    n3x3=256,
        #                                    n5x5red=24,
        #                                    n5x5=64,
        #                                    pool_planes=64)
        # self.inception4D = InceptionBlock(in_channels=512,
        #                                    n1x1=112,
        #                                    n3x3red=144,
        #                                    n3x3=288,
        #                                    n5x5red=32,
        #                                    n5x5=64,
        #                                    pool_planes=64)
        # self.inception4E = InceptionBlock(in_channels=528,
        #                                    n1x1=256,
        #                                    n3x3red=160,
        #                                    n3x3=320,
        #                                    n5x5red=32,
        #                                    n5x5=128,
        #                                    pool_planes=128)
        self.inception4A = InceptionBlock(480, 192, 96, 208, 16, 48, 64, 'inception_4a', weights)
        self.inception4B = InceptionBlock(512, 160, 112, 224, 24, 64, 64, 'inception_4b', weights)
        self.inception4C = InceptionBlock(512, 128, 128, 256, 24, 64, 64, 'inception_4c', weights)
        self.inception4D = InceptionBlock(512, 112, 144, 288, 32, 64, 64, 'inception_4d', weights)
        self.inception4E = InceptionBlock(528, 256, 160, 320, 32, 128, 128, 'inception_4e', weights)
        
        self.pool5 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        # self.inception5A = InceptionBlock(in_channels=832,
        #                                    n1x1=256,
        #                                    n3x3red=160,
        #                                    n3x3=320,
        #                                    n5x5red=32,
        #                                    n5x5=128,
        #                                    pool_planes=128)
        # self.inception5B = InceptionBlock(in_channels=832,
        #                                    n1x1=384,
        #                                    n3x3red=192,
        #                                    n3x3=384,
        #                                    n5x5red=48,
        #                                    n5x5=128,
        #                                    pool_planes=128)
        self.inception5A = InceptionBlock(832, 256, 160, 320, 32, 128, 128, 'inception_5a', weights)
        self.inception5B = InceptionBlock(832, 384, 192, 384, 48, 128, 128, 'inception_5b', weights)
        
        # The final classifier with the LossHeader
        self.loss_header = LossHeader()
        
        # Two auxiliary paths
        self.aux4A = InceptionAux(512) 
        self.aux4D = InceptionAux(528)    

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.inception3A(x)
        x = self.inception3B(x)
        x = self.pool4(x)
        x = self.inception4A(x)
  
        loss1_xyz, loss1_wpqr = self.aux4A(x)
        
        x = self.inception4B(x)
        x = self.inception4C(x)
        x = self.inception4D(x)
  
        loss2_xyz, loss2_wpqr = self.aux4D(x)
        
        x = self.inception4E(x)
        x = self.pool5(x)
        x = self.inception5A(x)
        x = self.inception5B(x)

        loss3_xyz, loss3_wpqr = self.loss_header(x)

        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):
    def __init__(self, w1, w2, w3, beta1, beta2, beta3):
        super(PoseLoss, self).__init__()
        
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr Quaternion

        # Position loss
        loss_xyz_1 = F.mse_loss(p1_xyz, poseGT[:, :3])
        loss_xyz_2 = F.mse_loss(p2_xyz, poseGT[:, :3])
        loss_xyz_3 = F.mse_loss(p3_xyz, poseGT[:, :3])

        # Orientation loss (quaternion)
        loss_wpqr_1 = F.mse_loss(p1_wpqr, poseGT[:, 3:])
        loss_wpqr_2 = F.mse_loss(p2_wpqr, poseGT[:, 3:])
        loss_wpqr_3 = F.mse_loss(p3_wpqr, poseGT[:, 3:])

        # Combine losses for each header
        loss = self.w1 * (loss_xyz_1 + self.beta1 * loss_wpqr_1) + \
                     self.w2 * (loss_xyz_2 + self.beta2 * loss_wpqr_2) + \
                     self.w3 * (loss_xyz_3 + self.beta3 * loss_wpqr_3)

        return loss




# for debugging

# Function to create a simple test dataset
# def create_test_dataset(num_samples=10, image_size=(3, 224, 224)):
#     images = torch.randn(num_samples, *image_size)  # Random images
#     poses = torch.randn(num_samples, 7)  # Random pose vectors
#     return Data.TensorDataset(images, poses)

# # # Function to test the PoseNet implementation
# def test_posenet():
#     # Parameters
#     num_samples = 10
#     image_size = (3, 224, 224)

#     # Create test dataset
#     test_dataset = create_test_dataset(num_samples, image_size)
#     test_loader = Data.DataLoader(dataset=test_dataset, batch_size=2)

#     # Initialize PoseNet and PoseLoss
#     posenet = PoseNet(load_weights=False)
#     criterion = PoseLoss(0.3, 0.3, 1., 300, 300, 300)

#     # Test the model with one batch of data
#     for images, poses in test_loader:
#         # Forward pass through PoseNet
#         outputs = posenet(images)

#         # Compute the loss
#         loss = criterion(*outputs, poses)

#         # Print results
#         print("Outputs:", outputs)
#         print("Loss:", loss.item())

#         # For simplicity, break after the first batch
#         break

# if __name__ == '__main__':
#     test_posenet()