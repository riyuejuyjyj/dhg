import torch
import dhg
from dhg.data import Cooking200

# 看看能否成功加载数据并移动到 GPU
data = Cooking200()
hg = dhg.Hypergraph(data['num_vertices'], data['edge_list'])
print(f"Hypergraph created with {hg.num_v} vertices.")

# 检查显存占用情况
print(f"Current device: {torch.cuda.get_device_name(0)}")