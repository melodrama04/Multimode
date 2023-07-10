import os
import json
import chardet
from tqdm import tqdm
from PIL import Image

# 将文本和标签格式化成一个json
def data_format(input_path, data_dir, output_path):
    data = []
    with open(input_path) as f:
        for line in tqdm(f.readlines(), desc='----- [Formating] -----'):
            guid, label = line.replace('\n', '').split(',')
            text_path = os.path.join(data_dir, (guid + '.txt'))
            if guid == 'guid': continue
            with open(text_path, 'rb') as textf:
                text_byte = textf.read()
                encode = chardet.detect(text_byte)
                try:
                    text = text_byte.decode(encode['encoding'])
                except:
                    try:
                        text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                    except:
                        print('not is0-8859-1', guid)
                        continue
            text = text.strip('\n').strip('\r').strip(' ').strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
    with open(output_path, 'w') as wf:
        json.dump(data, wf, indent=4)

# 读取数据，返回[(guid, text, img, label)]元组列表
def read_from_file(path, data_dir, only=None):
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='----- [Loading] -----'):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid': continue

            if only == 'text': img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))
                # img = cv2.imread(img_path)
                img = Image.open(img_path)
                img.load()

            if only == 'img': text = ''

            data.append((guid, text, img, label))
        f.close()

    return data