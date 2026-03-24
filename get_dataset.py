from dhg.data import Cora
import torch
import pandas as pd

# 1. 加载数据集
data = Cora()

# 2. 获取数据 (返回的是 PyTorch Tensor)
features = data['features']
labels = data['labels']
edge_list = data['edge_list'] # 原始引用边对
train_mask = data['train_mask']
test_mask = data['test_mask']

# 3. 导出为本地文件
# 导出为 PyTorch 标准格式 (最推荐，方便以后读取)
torch.save({
    'features': features,
    'labels': labels,
    'edge_list': edge_list,
    'train_mask': train_mask,
    'test_mask': test_mask
}, 'cora_dataset.pt')

# 或者导出特征矩阵为 CSV (如果需要用 Excel 或其他分析工具查看)
pd.DataFrame(features.numpy()).to_csv('cora_features.csv', index=False)

print("数据集已成功导出至本地！")