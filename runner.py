
# --------------------------
# 1. Download data (KaggleHub)
# --------------------------
import kagglehub

# Téléchargement des données
dataset_path = kagglehub.dataset_download('saifkhichi96/bank-checks-signatures-segmentation-dataset')
print('Data source import complete.')

# --------------------------
# 2. Dataset Definition
# --------------------------
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class ChecksDataset(Dataset):
    """Bank checks dataset."""
    def __init__(self, root, imsize=(512,512), transform=None):
        self.root = root
        self.clases = np.array(['bg', 'sign'])
        self.imsize = imsize
        self.transform = transform

        self.files = []
        Xs = sorted(os.listdir(os.path.join(self.root, 'X')))
        ys = sorted(os.listdir(os.path.join(self.root, 'y')))
        for im in Xs:
            im_clean = im[2:]
            if ('y_' + im_clean) in ys:
                self.files.append({
                    'image': os.path.join(self.root, 'X', 'X_' + im_clean),
                    'label': os.path.join(self.root, 'y', 'y_' + im_clean),
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]

        # Load image
        image = Image.open(data_file['image']).convert('RGB')
        image = image.resize(self.imsize)
        image = np.array(image, dtype=np.uint8).transpose(2, 0, 1)  # HWC → CHW

        # Load label
        label = Image.open(data_file['label']).convert('1')
        label = label.resize(self.imsize)
        label = np.array(label, dtype=np.int32)
        label = label.reshape(1, *label.shape)  # (1, H, W)

        return torch.tensor(image, dtype=torch.float32) / 255.0, torch.tensor(label, dtype=torch.float32)

# --------------------------
# 3. DataLoader Setup
# --------------------------
cuda = torch.cuda.is_available()
print('cuda:', cuda)

data_dir = 'kaggle/input/'
train_data = ChecksDataset(os.path.join(data_dir, 'TrainSet/'))
valid_data = ChecksDataset(os.path.join(data_dir, 'TestSet/'))

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, **kwargs)

print('The dataset contains %d training and %d test samples' % (len(train_loader), len(valid_loader)))

# --------------------------
# 4. Display Sample
# --------------------------
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def apply_mask(im, mask=None):
    im = im.numpy().transpose(1, 2, 0)  # CHW → HWC
    if mask is not None:
        mask = mask.numpy()[0]  # (1, H, W) → (H, W)
        im[mask != 0] = [255, 0, 0]  # Red
    return im.astype(np.uint8)

im, mask = next(iter(train_loader))

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title("Image")
plt.imshow(im[0].permute(1, 2, 0))  # CHW → HWC
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title("Image + Mask")
plt.imshow(apply_mask(im[0], mask[0]))
plt.axis('off')

plt.tight_layout()
plt.show()
