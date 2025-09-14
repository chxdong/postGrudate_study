import pandas as pd
from matplotlib import pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows = 1000)

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

# _ = plt.hist(train_df['text_len'], bins=200)
# plt.xlabel('Text char count')
# plt.title("Histogram of char count")
# plt.show()

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()

# 字符分布统计
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])

print(word_count[-1])
print('-------------------')

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])

print(word_count[1])

print(word_count[2])

# 假设3750, 900, 648是句子标点符号的编码
# 需要修改count_sentence的计算方式
def count_sentences(text):
    sentence_count = 0
    chars = text.split(' ')
    for i in chars:
        if i in ['3750', '900', '648']:
            sentence_count += 1
    return sentence_count




# 应用函数计算每篇新闻的句子数
train_df['count_sentence'] = train_df['text'].apply(count_sentences)

# 计算平均每篇新闻的句子数
average_sentences = train_df['count_sentence'].mean()
print(f"每篇新闻平均句子数: {average_sentences}")

# 统计每类新闻中出现次数最多的字符
def get_most_frequent_char_by_category():
    # 获取所有新闻类别
    categories = train_df['label'].unique()

    # 为每个类别统计字符频率
    for category in categories:
        # 筛选当前类别的新闻
        category_df = train_df[train_df['label'] == category]

        # 合并当前类别所有新闻的文本
        all_lines = ' '.join(list(category_df['text']))

        # 统计字符频率
        char_count = Counter(all_lines.split(" "))
        char_count = sorted(char_count.items(), key=lambda d: int(d[1]), reverse=True)

        # 输出该类别中出现次数最多的字符
        if char_count:
            print(f"类别 {category} 中出现次数最多的字符: {char_count[0]}")

# 调用函数
get_most_frequent_char_by_category()

# 基于机器学习的文本分类
# Count Vectors + RidgeClassifier
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

# 0.74
# TF-IDF +  RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.87


