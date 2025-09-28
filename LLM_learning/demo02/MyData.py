from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    # 初始化数据
    def __init__(self, split):
        # 从磁盘加载数据
        self.data = load_from_disk(r"D:\postGrudate\LLM_learning\dem02\data")
        