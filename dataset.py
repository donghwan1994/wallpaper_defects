import os
import glob
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class_name = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', 
    '면불량', '몰딩수정', '반점', '석고수정', '오염', '오타공', '울음', 
    '이음부불량', '창틀,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]


def get_data_list(data_dir, test_size=0.1, seed=41):
    all_img_list = glob.glob(os.path.join(data_dir, "train", "*", "*.png"))
    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])
    train, val, _, _ = train_test_split(df, df['label'], test_size=test_size, stratify=df['label'], random_state=seed)
    
    le = preprocessing.LabelEncoder()
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])
    
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test = [os.path.join(data_dir, img_path) for img_path in test['img_path'].values]
    
    return train, val, test


class WallPaper(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_path_list)

    def get_labels(self):
        return self.label_list
    
    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        image = Image.open(image_path)
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image


if __name__ == '__main__':
    data_dir = "/workspace/dataset/dacon_wallpaper_defects"
    tr, val, ts = get_data_list(data_dir=data_dir, test_size=0.2, seed=41)
    print(len(tr['img_path']))
    print(len(val['img_path']))
    print(len(ts))