from torch.utils.data import Dataset
import os
from utils import read_pts, read_seg
from transformers import PointSampler


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, is_train=False, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.files = []

        images_dir = root_dir + '02691156/points/'
        categories_dir = root_dir + '02691156/expert_verified/points_label/'

        for f in os.listdir(categories_dir):
            self.files.append({
                'pointcloud_path': images_dir + f.replace('.seg', '.pts'),
                'category_path': categories_dir + f,
            })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with open(self.files[idx]['pointcloud_path'], 'r') as f:
            pointcloud = read_pts(f)
        with open(self.files[idx]['category_path'], 'r') as f:
            category = read_seg(f)
        pointcloud, category = PointSampler(1650)((pointcloud, category))
        if self.is_train:
            pointcloud, category = self.transform((pointcloud, category))
        
        return pointcloud, category