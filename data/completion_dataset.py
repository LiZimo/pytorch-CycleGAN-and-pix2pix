import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import imageio
import torch

class CompletionDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.mask_path = sorted(make_dataset(opt.mask_path, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        #self.size = opt.load_size
        #self.size = 512
    def exr2rgb(self, tensor):
        return (tensor*12.92) * (tensor<=0.0031308) + (1.055*(torch.pow(tensor,(1.0/2.4)))-0.055) * (tensor>0.0031308)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        full = torch.tensor(imageio.imread(AB_path), dtype = torch.float32)
        full = torch.clip(full, 0. ,1,)
        full = self.exr2rgb(full)
        mean_color = torch.mean(full, dim = [0,1])
        mean_color_mat = torch.repeat(mean_color,512,512,1)

        partial = torch.clone(full)
        random_mask_id = random.randint(0, len(self.mask_path) - 1)
        mask = torch.tensor(imageio.imread(self.mask_path[random_mask_id])[:,:,np.newaxis].astype(np.float32), dtype = torch.float32)
        mask = torch.clip(mask, 0.,1.)
       # mask = mask[:,:,np.newaxis]
        mask = torch.cat((mask,mask,mask), axis = 2)
        #print(mask.min())
        #print(mask.max())
        #mask = cv2.resize(mask, (self.size, self.size))
        #mask = np.array(mask)

        #print(partial.shape)
        #print(full.shape)
        #print(mask.shape)

        #partial[mask == 0] = mean_color
        #mean_color = torch.mean(partial[partial!=0])
        #print(mean_color)
        partial[mask == 0] = 0
        A = torch.cat((partial, mask[:,:,0:1]), axis = 2)
        #A = partial
        A = A.permute(2,0,1)
        full = full.permute(2,0,1)
        #A = Image.fromarray(A, mode = 'RGBA')
        #B = Image.fromarray(full, mode = 'RGB')

        # split AB image into A and B
        #w, h = AB.size
        #w2 = int(w / 2)
        #A = AB.crop((0, 0, w2, h))
        #B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (A.shape[0], A.shape[1]))
        #print(transform_params)
        #transform_params = get_params(self.opt, (A.shape[0], A.shape[1]))
        A_transform = get_transform(self.opt, transform_params, num_channels = 3, grayscale=(self.input_nc == 1), convert = False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), convert = False)

        A = A_transform(A)
        B = B_transform(full)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
