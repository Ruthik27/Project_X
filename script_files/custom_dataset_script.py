import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DisasterImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and csv files.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the pre-disaster image and its path
        pre_image_name = self.annotations.iloc[idx, 0].replace('post', 'pre')
        pre_image_path = os.path.join(self.root_dir, 'pre', pre_image_name)
        pre_image = Image.open(pre_image_path).convert('RGB')
        
        # Get the post-disaster image and its path
        post_image_name = self.annotations.iloc[idx, 0]
        post_image_path = os.path.join(self.root_dir, 'post', post_image_name)
        post_image = Image.open(post_image_path).convert('RGB')

        # Get the corresponding text prompt
        prompt = self.annotations.iloc[idx, 1]
        
        # Apply transformations if any
        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)

        return {'pre_image': pre_image, 'post_image': post_image, 'prompt': prompt}
