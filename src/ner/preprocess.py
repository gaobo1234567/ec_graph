# import os
#
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, data_files
from transformers import AutoTokenizer

from configuration.config import *

def process():
    # 1. 读取数据（格式为字典），由于只有一个数据文件，所以key是train
    dataset = load_dataset('json', data_files=RAW_DATA_FILE)['train']
    # print(dataset)
    # Dataset({
    #     features: ['text', 'id', 'label', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'],
    #     num_rows: 1000
    # })

    # 2. 去除多余列,只保留text和label两列
    dataset = dataset.remove_columns(['id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'])

    # 3. 划分数据集，一次只能一分为二，所以需要分两次，训练集：测试集：验证集=8：1：1
    #先把训练集划分出去dataset_dict['train']
    dataset_dict = dataset.train_test_split(test_size=0.2)
    #再把刚才剩下的测试集对半分为测试集和验证集，注意这里为了避免命名错误要直接取value
    dataset_dict['test'], dataset_dict['valid'] = dataset_dict['test'].train_test_split(test_size=0.5).values()

    # 4. 定义分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 数据编码（输入文本和标签）
    def encode(example):
        # 5.1 将文本数据转成字符列表,因为NER任务需要对每一个token都输出一个概率。
        tokens = list( example['text'] )

        # 5.2 文本编码，is_split_into_words代表词列表已经被提前分好，
        # 只需让 tokenizer 再转换成模型能接受的输入格式：input_ids，token_type_ids，attention_mask
        # truncation表示如果输入太长，超过模型最大长度，就自动截断。
        # 这里不需要加padding，因为无法进行批处理
        inputs = tokenizer(tokens, is_split_into_words=True, truncation=True)

        # 5.3 进行实体标注
        entities = example['label']
        # 定义标注列表，存放id，默认都为‘O’的id
        labels = [ LABELS.index('O') ] * len(tokens)
        # 遍历每个Tag，标记为‘B’和‘I’的id
        for entity in entities:
            start = entity['start']
            end = entity['end']
            labels[start:end] = [LABELS.index('B')] + [LABELS.index('I')] * (end - start - 1)
        # 前后加上id=-100（默认的padding id不参与梯度计算），对应CLS和SEP，
        # 因为tokens经过tokenizer的处理（实际上是Bert），头尾会增加CLS和SEP，而labels要和处理之后的inputs长度保持一致，所以也得加
        labels = [-100] + labels + [-100]
        inputs['labels'] = labels
        return inputs

    #这里默认batch=false，因为每一条数据text长度不一致，label数量也不一致，无法分批处理
    dataset_dict = dataset_dict.map(encode, remove_columns=['text', 'label'] )
    print(dataset_dict['train'][0])
    # {'input_ids': [101, 130, 130, 4382, 4456, 5709, 3338, 3204, 2255, 1398, 1814, 677, 3862, 6205, 2128, 5722, 2336,
    #                7831, 5709, 6862, 6853, 3187, 7234, 1298, 776, 4767, 2157, 2411, 1760, 6414, 6843, 5709, 102],
    #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0],没有AB子句
    #  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                     1],无填充
    #  'labels': [-100, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2,
    #             -100]}

    # 6. 保存到文件夹，自动分为三个目录
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

if __name__ == '__main__':
    process()