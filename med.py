# coding:utf-8
# 和鲸社区医学数据挖掘算法评测大赛


import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import numpy as np


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
    wordA = data[data.category_A == 1].words
    wordB = data[data.category_B == 1].words
    wordC = data[data.category_C == 1].words
    wordD = data[data.category_D == 1].words
    wordE = data[data.category_E == 1].words
    wordF = data[data.category_F == 1].words
    print(wordA, wordB, wordC, wordD, wordE, wordF)
    
    
# 工具函数 判断字符串是否是数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except(TypeError, ValueError):
        pass
    return False

    
# 数据处理
def data_process(data):
    # 将句子分词，去除停用词
    with open("stopwords.txt", "r") as f:
        s = f.readlines()
    stopwords = pd.DataFrame()
    stop_list = []
    for c in s:
        stop_list.append(c[:-1])
    data["words"] = data["Question Sentence"].apply(lambda x : [i for i in jieba.cut(x) if (i not in stop_list) and not (is_number(i))])
    # print(data.head())
    return data
    
    
# 工具函数，将分词结果转换成字符串列表
def toStrList(X):
    result = []
    elem = []
    for i in X:
        s = " ".join(i)
        elem.append(s)
        result.append(s)
    return result
    
    
# 工具函数，词语向量化
def vecWords(words):
    # 词语向量化
    max_df = 0.8
    min_df = 3
    vect = CountVectorizer(max_df = max_df, min_df = min_df, token_pattern = u'(?u)\\b[^\\d\\W]\\w+\\b')
    x_words = toStrList(words)
    term = vect.fit_transform(x_words)
    term_matrix = pd.DataFrame(term.toarray(), columns = vect.get_feature_names())
    return (vect, term_matrix)
    
    
# 具体建模过程
def doModeling(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
    nb = MultinomialNB()
    vect, term_matrix = vecWords(x_train)
    pipe = make_pipeline(vect, nb)
    # print(pipe.steps)
    score = cross_val_score(pipe, toStrList(x_train), y_train, cv = 5, scoring = 'accuracy').mean()
    print("模型评分:", score)
    pipe.fit(toStrList(x_train), y_train)
    y_pred = pipe.predict(toStrList(x_test))
    print("准确率:", metrics.accuracy_score(y_test, y_pred))
    print("混淆矩阵:", metrics.confusion_matrix(y_test, y_pred))
    print("F1Score:", metrics.f1_score(y_test, y_pred, average='macro'))
    return pipe.predict
    
    
# 进行建模
def modeling(data):
    x = data.words
    words = toStrList(x)
    f = open("words.txt", "w")
    for word in words:
        f.write(word)
    f.close()
#    print(type(x.values), x.head(), x.shape)
#    term = vect.fit_transform(x.values)
#    term_matrix = pd.DataFrame(term, columns = vect.get_feature_names())
#    print(term_matrix.head())
    yA = data.category_A
    yB = data.category_B
    yC = data.category_C
    yD = data.category_D
    yE = data.category_E
    yF = data.category_F
    #x_train, x_test, yA_train, yA_test = train_test_split(x, yA, random_state = 1)

#    x_train, x_test, yB_train, yB_test = train_test_split(x, yB, random_state = 1)
#    x_train, x_test, yC_train, yC_test = train_test_split(x, yC, random_state = 1)
#    x_train, x_test, yD_train, yD_test = train_test_split(x, yD, random_state = 1)
#    x_train, x_test, yE_train, yE_test = train_test_split(x, yE, random_state = 1)
#    x_train, x_test, yF_train, yF_test = train_test_split(x, yF, random_state = 1)
    predA = doModeling(x, yA)
    predB = doModeling(x, yB)
    predC = doModeling(x, yC)
    predD = doModeling(x, yD)
    predE = doModeling(x, yE)
    predF = doModeling(x, yF)
    return (predA, predB, predC, predD, predE, predF)


# 用训练好的模型进行预测，输出规定格式的结果
def makeResult(filename, predict):
    data = pd.read_csv(filename)
    data = data_process(data)
    # print(data.head())
    words = toStrList(data.words)
    # print(words)
    results = pd.DataFrame()
    results["ID"] = data.ID
    results["category_A"] = predict[0](words)
    results["category_B"] = predict[1](words)
    results["category_C"] = predict[2](words)
    results["category_D"] = predict[3](words)
    results["category_E"] = predict[4](words)
    results["category_F"] = predict[5](words)
    print(results.head())
    results.to_csv("result.csv", index = None)
    print("输出完毕")
    
    
# 计算F1_Score值
def f1_score(inputFile, outputFile):
    input = pd.read_csv(inputFile)
    output = pd.read_csv(outputFile)
    f1_A = metrics.f1_score(input.category_A, output.category_A, average='macro')
    f1_B = metrics.f1_score(input.category_B, output.category_B, average='macro')
    f1_C = metrics.f1_score(input.category_C, output.category_C, average='macro')
    f1_D = metrics.f1_score(input.category_D, output.category_D, average='macro')
    f1_E = metrics.f1_score(input.category_E, output.category_E, average='macro')
    f1 = np.mean([f1_A, f1_B, f1_C, f1_D, f1_E])
    return f1
    

if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv("train.csv")
    data = data_process(data)
    analysis(data)
    predict = modeling(data)
    makeResult("train.csv", predict)
    f1 = f1_score("train.csv", "result.csv")
    print("f1分值:", f1)
    