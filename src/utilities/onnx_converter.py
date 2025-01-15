from onnxsim import simplify
import onnx
import torch.onnx
from face_detection import *
from distance_detection import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from DAV2.metric_depth.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn.functional as F
import torchvision
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import random

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, 
                     padding='same', bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Optional projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Reshape to (batch, seq_len=1, channels)
        x = x.unsqueeze(1)
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Apply attention
        out, _ = self.attention(q, k, v)
        return out.squeeze(1)

def make_net(output_dim=1):
    act = lambda: nn.GELU()
    bn = lambda ch: nn.BatchNorm2d(ch)
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=2, padding=0, bias=True),
            act(),
            nn.Dropout2d(0.1),  # Light dropout in early layers
        ),
        nn.Sequential(
            conv(24, 64),
            nn.MaxPool2d(2),
            bn(64), act(),
            nn.Dropout2d(0.2),
            conv(64, 64),
            bn(64), act(),
            nn.Dropout2d(0.2),
        ),
        nn.Sequential(
            conv(64, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            nn.Dropout2d(0.3),
            conv(256, 256),
            bn(256), act(),
            nn.Dropout2d(0.3),
            ChannelAttention(256),
        ),
        nn.Sequential(
            conv(256, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            nn.Dropout2d(0.4),
            conv(256, 256),
            bn(256), act(),
            nn.Dropout2d(0.4),
        ),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Dropout(0.5),  # Regular dropout before final linear layer
        nn.Linear(256, output_dim, bias=False),
        Mul(1/9)
    )

class plmodel(pl.LightningModule):
    def __init__(self, model, num_classes, lr, num_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        logits = logits.squeeze(1)
        y = y.float()
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        pred_class = (torch.sigmoid(logits) > 0.5).float()
        self.train_acc.update(pred_class, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        logits = logits.squeeze(1)
        y = y.float()
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
    
        pred_class = (torch.sigmoid(logits) > 0.5).float()
        self.val_acc.update(pred_class, y)
        
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        logits = logits.squeeze(1)
        y = y.float()
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        pred_class = (torch.sigmoid(logits) > 0.5).float()
        self.test_acc.update(pred_class, y)
        
        self.log('test_loss', loss, prog_bar=True)

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.90, 0.95), weight_decay=0.01)
        
        # Create warmup scheduler
        warmup_steps = int(0.1 * self.hparams.num_epochs)  # 10% of total epochs for warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of max lr
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create decay scheduler
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.hparams.num_epochs - warmup_steps
        )
        
        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

class Resize(object):
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False,
                 ensure_multiple_of=1, resize_method="lower_bound",
                 image_interpolation_method=cv2.INTER_AREA):
        # ... initialization stays the same ...
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        # Convert to numpy for calculation then back to tensor
        y = (torch.round(x / self.__multiple_of) * self.__multiple_of).int()
        
        if max_val is not None and y > max_val:
            y = (torch.floor(x / self.__multiple_of) * self.__multiple_of).int()
            
        if y < min_val:
            y = (torch.ceil(x / self.__multiple_of) * self.__multiple_of).int()
            
        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(torch.tensor(scale_height * height), min_val=self.__height)
            new_width = self.constrain_to_multiple_of(torch.tensor(scale_width * width), min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(torch.tensor(scale_height * height), max_val=self.__height)
            new_width = self.constrain_to_multiple_of(torch.tensor(scale_width * width), max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(torch.tensor(scale_height * height))
            new_width = self.constrain_to_multiple_of(torch.tensor(scale_width * width))
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (int(new_width), int(new_height))

    # get_size method remains largely the same, just using torch operations
    
    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
        # Resize using torch interpolate instead of cv2
        image = sample["image"].permute(2, 0, 1).unsqueeze(0)  # Convert to [1, C, H, W]
        image = F.interpolate(image, size=(height, width), mode='bicubic', align_corners=False)
        sample["image"] = image.squeeze(0).permute(1, 2, 0)  # Back to [H, W, C]
        
        if self.__resize_target:
            if "depth" in sample:
                depth = sample["depth"].unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, size=(height, width), mode='nearest')
                sample["depth"] = depth.squeeze(0).squeeze(0)
                
            if "mask" in sample:
                mask = sample["mask"].unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(height, width), mode='nearest')
                sample["mask"] = mask.squeeze(0).squeeze(0)
        
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.__mean = torch.tensor(mean)
        self.__std = torch.tensor(std)

    def __call__(self, sample):
        # Ensure mean and std are on the same device as the image
        mean = self.__mean.to(sample["image"].device)
        std = self.__std.to(sample["image"].device)
        
        # Normalize
        sample["image"] = (sample["image"] - mean) / std
        return sample


class PrepareForNet(object):
    def __call__(self, sample):
        # Convert from [H, W, C] to [C, H, W]
        sample["image"] = sample["image"].permute(2, 0, 1)
        
        if "depth" in sample:
            sample["depth"] = sample["depth"].float()
        
        if "mask" in sample:
            sample["mask"] = sample["mask"].float()
        
        return sample


def image2tensor(raw_image, input_size=518):
    # Assuming raw_image is a torch tensor in [H, W, C] format
    h, w = raw_image.shape[:2]
    
    # Normalize to 0-1 range
    image = raw_image / 255.0
    
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    image = transform({'image': image})['image']
    # At this point image is [C, H, W] from PrepareForNet
    image = image.unsqueeze(0)  # Add batch dimension to make it [1, C, H, W]
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = image.to(DEVICE)
    
    return image, (h, w)

def optimize_model(model, sample_input: dict, dynamic_axes: dict, onnx_path: str):
    input_tuple = tuple(sample_input.values())

    torch.cuda.empty_cache()

    torch.onnx.export(
        model,              
        input_tuple,       
        onnx_path,     
        export_params=True,  
        opset_version=18,   
        do_constant_folding=True, 
        input_names=list(sample_input.keys()),  
        output_names=['output'],  
        dynamic_axes=dynamic_axes,
    )

    onnx.checker.check_model(onnx_path)
    print(f"Model successfully exported to ONNX at {onnx_path}")
    simplified_model, check = simplify(onnx_path)
    if check:
        onnx.save(simplified_model, onnx_path)
    print(f"Successfully optimized ONNX model")

class depth_wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_dav2()
        
    def forward(self, raw_image): 
        image, (h, w) = image2tensor(raw_image, 518)
        depth = self.model(image)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth   
       

# Create and export model
# model = depth_wrapper()
# # Create sample input as numpy array (H,W,C)
# sample = torch.randn((518, 518, 3))

# sample_input = {'input': sample}
# dynamic_shapes = {'input': {0: 'height', 1: 'width', 2: 'channels'}}
# optimize_model(model, sample_input, dynamic_shapes, 'dav2.onnx')
# Load model
base_model = make_net()
model = plmodel.load_from_checkpoint(
    checkpoint_path = 'squinting/checkpoints/squint_detector-epoch=1499-val_acc=0.94.ckpt',
    model=make_net(),
)
model.eval().cuda()

sample = torch.randn((1, 1, 256, 256)).cuda()
sample_input = {'input': sample}
dynamic_shapes = {'input': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}}
optimize_model(model, sample_input, dynamic_shapes, 'squint_detector.onnx')
