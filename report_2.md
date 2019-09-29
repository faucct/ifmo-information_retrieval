# Индексирование коллекции BY.WEB

## Индексация

Параллельная индексация всех партиций заняла 2 минуты, в то время как последовательная заняла 3 минуты.
Размер полученного индекса составляет 1.5 gb.

## Оценка качества поиска

Выполнение данных поисковых запросов по одному занимает 2.5 секунды.
Выполнение запросов пачками по 10 или 100 запросов позволяет уменьшить это время до 1.4 секунд.

Посчитанные метрики:

| Метрика              | Среднее | Медиана |
|----------------------|---------|---------|
| Precision@20         | 0.30    | 0.2     |
| Recall@20            | 0.22    | 0.17    |
| Average precision@20 | 0.63    | 0.68    |
| Recall-precision@20  | 0.19    | 0.16    |


## Оценка качества с PageRank.

PageRank был посчитан в прошлом задании для построения графа гиперссылок.
Значения хранились в виде словаря по URL ссылкам с числом ссылок на страницу. 
Затем все значения нормировались на общее число ссылок.
При индексации кроме выделенного, в прошлом задании, из документов текста использовались выделенные гиперссылки.
Так же использовался логарифмический скейлинг.


Как видно полученные метрики почти не отличаются от метрик без PageRank 
(отличия только в тысячных):

| Метрика              | Среднее | Медиана |
|----------------------|---------|---------|
| Precision@20         | 0.30    | 0.2     |
| Recall@20            | 0.22    | 0.17    |
| Average precision@20 | 0.63    | 0.68    |
| Recall-precision@20  | 0.19    | 0.16    |