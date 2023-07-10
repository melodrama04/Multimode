import os
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import torch
import argparse
from Utils import data_format, read_from_file
from DataProcess import Processor
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(inplace=True)
        )

    def forward(self, bert_inputs, masks, token_type_ids=None):
        # assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']

        return self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )

        self.trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.full_resnet.fc.in_features, 64),
            nn.ReLU(inplace=True)
        )

        # fine-tune
        for param in self.full_resnet.parameters():
                param.requires_grad = False

    def forward(self, imgs):
        feature = self.resnet(imgs)

        return self.trans(feature)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel()
        # image
        self.img_model = ImageModel()

        # 全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128,3),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.68, 9.3, 3.36]))

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_feature = self.text_model(texts, texts_mask)

        img_feature = self.img_model(imgs)

        text_prob_vec = self.text_classifier(text_feature)
        img_prob_vec = self.img_classifier(img_feature)
        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels

class Trainer():

    def __init__(self, processor, model,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.processor = processor
        self.model = model.to(device)
        self.device = device

        bert_params = set(self.model.text_model.bert.parameters())
        resnet_params = set(self.model.img_model.full_resnet.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'lr': 5e-6, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 5e-6, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'lr': 5e-6, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'lr': 5e-6, 'weight_decay': 0.0},
            {'params': other_params,
             'lr': 5e-6, 'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(params, lr=3e-5)

    def train(self, train_loader):
        self.model.train()

        loss_list = []
        true_labels, pred_labels = [], []

        for batch in tqdm(train_loader, desc='----- [Training] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)

            # metric
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list

    def valid(self, val_loader):
        self.model.eval()

        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(val_loader, desc='\t ----- [Validing] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)

            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

        metrics = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(val_loader), metrics

    def predict(self, test_loader):
        self.model.eval()
        pred_guids, pred_labels = [], []

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device)
            pred = self.model(texts, texts_mask, imgs)

            pred_guids.extend(guids)
            pred_labels.extend(pred.tolist())

        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]

# args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='训练模型')
parser.add_argument('--text_pretrained_model', default='bert-base-uncased', help='文本分析模型', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=10, help='设置训练轮数', type=int)

parser.add_argument('--test', action='store_true', help='预测测试集数据')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')
args = parser.parse_args()

learning_rate = args.lr
weight_decay = args.weight_decay
epoch = args.epoch
bert_name = args.text_pretrained_model

load_model_path = args.load_model_path
only = 'img' if args.img_only else None
only = 'text' if args.text_only else None
if args.img_only and args.text_only: only = None

print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format('bert-base-uncased', 'ResNet50', 'NaiveCombine'))

# Initilaztion
processor = Processor()
model = Model()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trainer = Trainer(processor, model, device)

root_path = root_path = os.getcwd()
data_dir = os.path.join(root_path, './data')
train_data_path = os.path.join(root_path, 'data/train.json')
test_data_path = os.path.join(root_path, 'data/test.json')
output_path = os.path.join(root_path, 'output')
output_test_path = os.path.join(output_path, 'test.txt')

checkout_params = {'batch_size': 4, 'shuffle': False}
train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
test_params = {'batch_size': 8, 'shuffle': False, 'num_workers': 2}
# Train
def train():

    data_format(os.path.join(root_path, 'train.txt'),
                os.path.join(root_path, './data'), os.path.join(root_path, './data/train.json'))
    data = read_from_file(train_data_path, data_dir, only)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, train_size=(0.8), test_size=0.2)
    train_loader = processor(train_data, train_params)
    val_loader = processor(val_data, val_params)

    best_acc = 0
    # epoch = epoch
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e + 1) + ' ' + '-' * 20)
        tloss, tloss_list = trainer.train(train_loader)
        print('Train Loss: {}'.format(tloss))
        vloss, vacc = trainer.valid(val_loader)
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))
        if vacc > best_acc:
            best_acc = vacc
            output_model_dir = os.path.join(output_path, 'NaiveCombine')
            if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)  # 没有文件夹则创建
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

            print('Update best model!')
        print()


# Test
def test():
    data_format(os.path.join(root_path, 'test_without_label.txt'),
                os.path.join(root_path, './data'), os.path.join(root_path, './data/test.json'))
    test_data = read_from_file(test_data_path, data_dir, only)
    test_loader = processor(test_data, test_params)

    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))

    outputs = trainer.predict(test_loader)
    formated_outputs = processor.decode(outputs)
    with open(output_test_path, 'w') as f:
        for line in tqdm(formated_outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()

# main
if __name__ == "__main__":
    if args.train:
        train()

    if args.test:
        test()