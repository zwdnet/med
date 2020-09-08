# coding:utf-8
# 和鲸社区医学数据挖掘算法评测大赛


import pandas as pd
import jieba


# 数据分析
def analysis(data):
    print(data.info())
    print(data.head())
    # 看看各分类的数量
    print(data[data.category_A == 1].category_A.count())
    print(data[data.category_B == 1].category_B.count())
    print(data[data.category_C == 1].category_C.count())
    print(data[data.category_D == 1].category_D.count())
    print(data[data.category_E == 1].category_E.count())
    print(data[data.category_F == 1].category_F.count())
    
    
# 数据处理
def data_process(data):
    data["words"] = data["Question Sentence"].apply(lambda x : list(jieba.cut(x)))
    print(data.head())
    return data


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv("train.csv")
    analysis(data)
    data = data_process(data)
    