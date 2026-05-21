from pathlib import Path

# 1. 目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / 'data'
NER_DIR = 'ner'
RAW_DATA_DIR = DATA_DIR / NER_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / 'processed'

LOG_DIR = ROOT_DIR / 'logs'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

# web 静态目录
WEB_STATIC_DIR = ROOT_DIR / 'src' / 'web' / 'static'

# 2. 数据文件名 和 模型名称
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')
MODEL_NAME = 'google-bert/bert-base-chinese'

# 3. 超参数
BATCH_SIZE = 2 #这个数据集太小，所以batch_size设置小一些，让模型多训练几次
EPOCHS = 5
LEARNING_RATE = 5e-5

SAVE_STEPS = 20

# 4. NER任务分类标签
LABELS = ['B', 'I', 'O']

# 5. 数据库连接
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'G20021004b',
    'database': 'gmall',
}

NEO4J_CONFIG = {
    'uri': "neo4j://localhost:7687",
    'auth': ("neo4j", "G20021004b")
}

# API_KEY = 'sk-b9ef2d89636e4c01adb30fe1b6c3739d'