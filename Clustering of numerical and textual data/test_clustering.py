#!/usr/bin/env python3

# задаём список файлов выборки
input_files = ['./dickens1.txt',
               './dickens2.txt',
               './dickens3.txt',
               './bukowski1.txt',
               './bukowski2.txt',
               './bukowski3.txt']

# применяем один из методов извлечения
# признаков из текста --- подсчёт слов
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(input='filename')
# рассчитываем признаки, выводим их количество
X_train_counts = count_vect.fit_transform(input_files)
print(X_train_counts.shape)

# применим один из методов кластеризации
from sklearn.cluster import KMeans
cl_alg = KMeans(n_clusters=2, n_init='auto')
# кластеризуем данные по признакам, выводим ярлыки
result = cl_alg.fit(X_train_counts)
print(result.labels_)
