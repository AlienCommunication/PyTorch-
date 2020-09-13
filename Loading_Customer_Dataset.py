#!/usr/bin/env python
# coding: utf-8

# ## How to transform Custom dataset into Tensors

# In[1]:


import torch
import os
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data  import dataset


# In[ ]:


"""
Let's prepare the dataset

Here Assume, You have some Image Dataset with some csv file. In this csv file.

"""

class CarBike(Dataset):
    
    """
    Arg:
    
    csv_file = This csv file will be having labels
    
    root_dir = Will have directory path of the images
    
    transform = it is options but we will use it later to transform the dataset into tensors
    
    """
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        
        if self.transform:
            image = self.transform(image)
            
            
        return (image, y_label)
    
    
dataset = CarBike(csv_file = "CarBike.csv", root_dir = "CarBikeImage", transform = transforms.ToTensor())


train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])


train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)





        
    
    

