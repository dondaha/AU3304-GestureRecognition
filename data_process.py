# data_process.py 数据预处理脚本
# 脚本从data/目录下读取数据，将处理后的输出保存到data_processed/目录下。

## 1. 确保目录存在
import os
os.makedirs('data_processed', exist_ok=True)

## 2. 