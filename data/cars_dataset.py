
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pandas as pd
import torch


def to_decade(year):
  new_year = round((float(year))/10)*10
  return new_year

def treat_df(df):
  new_df = df.applymap(to_decade)
  min_year = new_df.Year.min()
  max_year = new_df.Year.max()
  dic_label = {}
  for i,k in enumerate(range(min_year,max_year+1,10)):
    dic_label[k] = i

  new_df = new_df.applymap(lambda x: dic_label[x] )
  return new_df

class CarsDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.folder = opt.dataroot
        csv_path = os.path.join(self.folder,'year_dir.csv')
        self.csv_path = csv_path
        # self.image_size = image_size
        self.df = pd.read_csv(csv_path)
        self.df = treat_df(self.df)
        self.dim = self.df.Year.max()+1

        input_nc = self.opt.input_nc
        self.transforms = get_transform(self.opt, grayscale=(input_nc == 1))
        
        # transforms = []
        # # transforms.append(convert_transparent_to_rgb)
        # transforms.append(T.Resize((image_size,image_size)))
        # transforms.append(T.ToTensor())
        # self.transforms = T.Compose(transforms)
        
  

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        data_year = self.df.iloc[index].Year
        path = os.path.join(self.folder,str(index)+".jpg")
        img = Image.open(path)
        # if len(np.shape(img))<3 or np.shape(img)[2]==1 :
        #   print(index)
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)

        img_transform = self.transforms(rgbimg)
        # data_year = torch.tensor(data_year)
        data_year_one_hot = torch.zeros(self.dim).scatter_(0, torch.tensor([data_year]), 1.0)
        print("data_year_one_hot",data_year_one_hot.shape)
        return img_transform,data_year_one_hot



