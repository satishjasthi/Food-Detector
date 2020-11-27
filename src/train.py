import torch, shutil
from PIL import Image
import pandas as pd
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
# from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, LightningDataModule, LightningModule
from pytorch_lightning.metrics import Accuracy
from torchvision.transforms.transforms import Resize


class DataModule(LightningDataModule):

    def __init__(self, exp_args):
        super().__init__()
        self.num_classes = exp_args.num_classes 
        self.data_dir = Path(exp_args.data_dir)
        self.batch_size = exp_args.batch_size
        

    def prepare_data(self):
        raw_data_dir = self.data_dir/'raw_data'
        
        # remove all corrupt images
        print('Checking for corrupt images...............')
        num_corrupt = 0
        for img in list(raw_data_dir.glob('*/*')):
            try:
                img = Image.open(img)
            except:
                dest = self.data_dir/f'trash/{img.name}'
                shutil.move(img, dest)
                num_corrupt+1
        print(f"Moved {num_corrupt} corrupted images to data/trash dir")
        
        target_class = raw_data_dir/'white_rice'
        other_classes = [raw_data_dir/'idli', raw_data_dir/'dosa', raw_data_dir/'chapati']
        
        rows = [{'img_pth':str(file_name), 'class':'white_rice'} for file_name in list(target_class.glob('*'))]
        rows.extend([{'img_pth':str(file_name), 'class':'not_white_rice'} for file_name in list(other_classes[0].glob('*'))])
        rows.extend([{'img_pth':str(file_name), 'class':'not_white_rice'} for file_name in list(other_classes[1].glob('*'))])
        rows.extend([{'img_pth':str(file_name), 'class':'not_white_rice'} for file_name in list(other_classes[2].glob('*'))])
        all_imgs_df = pd.DataFrame(data=rows)
        
        # create positive and negative class data
        num_positive_samples = all_imgs_df[all_imgs_df['class']=='white_rice'].shape[0]
        negative_samples_df = all_imgs_df[all_imgs_df['class']=='not_white_rice'].sample(n=num_positive_samples,random_state=32)
        model_data = pd.concat([all_imgs_df[all_imgs_df['class']=='white_rice'], negative_samples_df], axis='rows').sample(frac=1)
        
        # create train and valid data
        train_dfs,valid_dfs = [],[],
        for class_name in model_data['class'].unique():
            class_df = model_data[model_data['class']==class_name].sample(frac=1)
            num_samples = class_df.shape[0]
            num_train_samples = int(round(num_samples*0.7))
            train_dfs.append(class_df.iloc[:num_train_samples,:])
            valid_dfs.append(class_df.iloc[num_train_samples:,:])
        train_data, valid_data =  pd.concat(train_dfs,axis='rows'), pd.concat(valid_dfs,axis='rows')
        
        train_dir = self.data_dir/'train'
        if not train_dir.exists():
            train_dir.mkdir(exist_ok=True, parents=True)
        val_dir =  self.data_dir/'valid'
        if not val_dir.exists():
            val_dir.mkdir(exist_ok=True, parents=True)
            
        print('Creating train data')
        print(f"Train data dist: {train_data['class'].value_counts()}")
        for _, row in train_data.iterrows():
            img_pth = Path(row['img_pth'])
            cls_name = row['class']
            cls_dir = train_dir/f'{cls_name}'
            if not cls_dir.exists():
                cls_dir.mkdir(exist_ok=True, parents=True)
            dest = cls_dir/f'{img_pth.name}'
            shutil.copy(img_pth, dest)
            
        print('Creating valid data')
        print(f"Valid data dist: {valid_data['class'].value_counts()}")
        for _, row in valid_data.iterrows():
            img_pth = Path(row['img_pth'])
            cls_name = row['class']
            cls_dir = val_dir/f'{cls_name}'
            if not cls_dir.exists():
                cls_dir.mkdir(exist_ok=True, parents=True)
            dest = cls_dir/f'{img_pth.name}'
            shutil.copy(img_pth, dest)

    def setup(self, stage):
        """
        create datasets for train and valid
        """
        train_dir = self.data_dir/'train'
        val_dir =  self.data_dir/'valid'
        
        self.train_transforms = transforms.Compose([transforms.Resize(256),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.val_transforms = transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        
        self.train_dataset = ImageFolder(str(train_dir), transform=self.train_transforms)
        self.val_dataset = ImageFolder(str(val_dir), transform=self.val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)
    

class Detectron(LightningModule):

    def __init__(self, exp_args):
        super().__init__()
        # save all hyper param config to check point
        self.num_classes = exp_args.num_classes
        self.lr = exp_args.lr
        self.accuracy = Accuracy()
        self.save_hyperparameters()
        
        # define pretained model
        pretrained_model = models.resnet18(pretrained=True)
        last_layer_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(last_layer_features, self.num_classes)
        self.model = pretrained_model        
    
        
    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        loss, logits, y = self.compute_loss(batch)
        return {'loss' : loss, 'preds' : logits, 'target' : y}
        
        
    def training_step_end(self, outputs):
        acc = self.accuracy(outputs['preds'], outputs['target'])
        self.log('train_loss', outputs['loss'],  logger=True)
        self.log('train_acc', acc, logger=True)
        
            
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.compute_loss(batch)
        return {'loss' : loss, 'preds' : logits, 'target' : y}
    
    def validation_step_end(self, outputs):
        acc = self.accuracy(outputs['preds'], outputs['target'])
        self.log('valid_loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        
def main(exp_args):
    # seed the entire experiment
    seed_everything(0)
    data = DataModule(exp_args)
    model = Detectron(exp_args)
    trainer = Trainer.from_argparse_args(exp_args)
    trainer.fit(model, data)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # add program level args
    parser.add_argument('--conda_env', type=str, default='foodie')
    parser.add_argument('--notification_email', type=str, default='satish27may@gmail.com')

    # add model specific args
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)

    exp_args = parser.parse_args()
    main(exp_args)
    #python train.py --conda_env foodie --notification_email satish27may@gmail.com --num_classes 2 --data_dir /home/satish27may/Foodie/data/ --batch_size 64 --gpus 1