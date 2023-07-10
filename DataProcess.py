from torch.utils.data import DataLoader

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from sklearn.metrics import accuracy_score

class OperateDataset(Dataset):

    def __init__(self, guids, texts, imgs, labels) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
               self.imgs[index], self.labels[index]

    def collate_fn(self, batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch])
        labels = torch.LongTensor([b[3] for b in batch])

        ''' 处理文本 统一长度 增加mask tensor '''
        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)

        return guids, paded_texts, paded_texts_mask, imgs, labels


class LabelVocab:
    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id.update({label: len(self.label2id)})
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label)

    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:
    def __init__(self) -> None:

        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        labelvocab = self.labelvocab

        labelvocab.add_label('positive')
        labelvocab.add_label('neutral')
        labelvocab.add_label('negative')
        labelvocab.add_label('null')  # 空标签

        ''' 文本处理 BERT的tokenizer '''
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        ''' 图像处理 torchvision的transforms '''

        def get_resize(image_size):
            for i in range(20):
                if 2 ** i >= image_size:
                    return 2 ** i
            return image_size

        img_transform = transforms.Compose([
            transforms.Resize(get_resize(224)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #RGB单个通道的值是[0, 255]，所以一个通道的均值应该在127附近才对。
            #如果Normalize()函数去计算 x = (x - mean)/std ，因为RGB是[0, 255]，算出来的x就不可能落在[-1, 1]区间了。
            #应用了torchvision.transforms.ToTensor，其作用是将数据归一化到[0,1]（是将数据除以255），transforms.ToTensor（）会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB）
        ])

        ''' 对读入的data进行预处理 '''
        guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
        for line in tqdm(data, desc='----- [Encoding]'):
            guid, text, img, label = line
            # id
            guids.append(guid)

            # 文本
            text.replace('#', '')
            tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
            encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))

            # 图像
            encoded_imgs.append(img_transform(img))

            # 标签
            encoded_labels.append(labelvocab.label_to_id(label))

        return guids, encoded_texts, encoded_imgs, encoded_labels

    def decode(self, outputs):
        labelvocab = self.labelvocab
        formated_outputs = ['guid,tag']
        for guid, label in tqdm(outputs, desc='----- [Decoding]'):
            formated_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
        return formated_outputs

    def metric(self, inputs, outputs):
        # print(classification_report(inputs, outputs))
        return accuracy_score(inputs, outputs)

    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return OperateDataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)