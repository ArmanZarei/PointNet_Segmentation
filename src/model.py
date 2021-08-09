import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        out = input.transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = nn.MaxPool1d(out.size(-1))(out)
        out = nn.Flatten(1)(out)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out).view(-1, self.k, self.k)

        init = torch.eye(self.k, requires_grad=True).repeat(input.size(0), 1, 1) 
        if out.is_cuda:
          init = init.cuda()
        out += init

        return out


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_transform = TNet(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, input):
        num_points = input.size(1)

        mat_3x3 = self.input_transform(input)
        out = torch.bmm(input, mat_3x3).transpose(1, 2)

        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = out.transpose(1, 2)
        mat_64x64 = self.feature_transform(out)
        local_point_features = torch.bmm(out, mat_64x64)
        out = local_point_features.transpose(1, 2)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))

        out = nn.MaxPool1d(out.size(-1))(out)

        out = out.view(-1, 1024, 1).repeat(1, 1, num_points).transpose(1, 2)
        out = torch.cat([local_point_features, out], 2)

        return out, mat_3x3, mat_64x64


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.transform = Transform()
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn4 = nn.BatchNorm1d(num_classes)

    def forward(self, input):
        out, mat_3x3, mat_64x64 = self.transform(input)
        out = out.transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        return F.log_softmax(out, dim=1), mat_3x3, mat_64x64