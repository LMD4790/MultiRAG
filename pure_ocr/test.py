import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#%%
# ==================== 示例1：基础折线图 ====================
def basic_line_chart():
    """
    基础折线图示例
    """
    # 创建数据
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [10, 15, 13, 17, 20, 25, 22]

    # 创建图形和坐标轴
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(x, y,
             marker='o',  # 数据点标记形状
             linestyle='-',  # 线条样式：实线
             color='blue',  # 线条颜色
             linewidth=2,  # 线条宽度
             markersize=8,  # 标记大小
             label='销售额')  # 图例标签

    # 设置标题和标签
    plt.title('产品销售额趋势图', fontsize=16, fontweight='bold')
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('销售额（万元）', fontsize=12)

    # 设置刻度
    plt.xticks(x, ['一月', '二月', '三月', '四月', '五月', '六月', '七月'])

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend()

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

basic_line_chart()

#%%

