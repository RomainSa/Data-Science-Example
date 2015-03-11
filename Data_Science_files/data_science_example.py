# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Introduction & Objective

# <markdowncell>

# Zalora is a online fashion retailer. We have hundreds of thousands of products accross all of our ventures -
# Singapore, Indonesia, Malaysia, Hong Kong, Thailand, Philippines, Vietnam, Brunei, Australia and New Zealand.
# One of our challenging problems, is to determine which products are better than others, essentially a venturewide
# ranking for all of its products. The ranking can be different for each different product category / subcategory.
# The result of this ranking serves a lot of purposes:
# Displaying best selling products on our homepage.
# Sorting our catalogs / search results whenever users surf our homepage.
# Sending better, more profitable email campaigns.

# <headingcell level=1>

# Dataset

# <markdowncell>

# (Attached as a single products.csv file) Luckily, our Data Engineering team has done a lot of data cleaning and
# aggregation to produce everything in a clearly formatted table. The given data set contains 4000 Indonesia
# products and their metadatas that are currently available on our store, sorted by a random order.
# [Notice: a large part of this was randomized since it contains confidential information]

# <headingcell level=1>

# Questions

# <headingcell level=3>

# 1) Give us your suggestions on how we could make our data set better / more useful.

# <markdowncell>

# First, let's import some packages, load the dataset and perform some treatments on the date:

# <codecell>

from math import log
from pandas import DataFrame
from datetime import datetime
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
from random import sample
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# <codecell>

dataFile = '/home/roms/Desktop/Zalora/data-science-zalora/products.csv'
data = DataFrame.from_csv(dataFile, encoding = 'ISO-8859-1')
data['activated_at'] = data['activated_at'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
quantitative = data.select_dtypes(include=['int64', 'float64']).columns
qualitative = data.select_dtypes(exclude=['int64', 'float64','datetime']).columns
other_date = data.select_dtypes(include=['datetime']).columns

# <markdowncell>

# We have a dataset with:
# * 4000 observations
# * 25 columns, of which:
#     * 6 are qualitative (ex: 'brand')
#     * 18 are quantitative (ex: 'original_price')
#     * 1 is a timestamp
# * no missing data (but field 'special_price' is left empty when there is no discount)

# <markdowncell>

# One thing we can do in order to improve the dataset is to calculate some indicators, such as:
# * the day of the year (1 being 1st of january) rather than a timestamp

# <codecell>

data['day'] = data['activated_at'].apply(lambda x : x.timetuple().tm_yday)

# <markdowncell>

# * the number of views per impression for each item (%viewed)

# <codecell>

data['%viewed'] = data['views_count']/data['impressions_count']

# <markdowncell>

# * the net number of items bought per view for each item (%bought)

# <codecell>

data['%bought'] = (data['net_sale_count']/data['views_count'])

# <markdowncell>

# * the discount rate for each item (%discount)

# <codecell>

data['special_price'] = data[['special_price','original_price']].min(axis=1)
data['%discount'] = (1-data['special_price']/data['original_price'])

# <markdowncell>

# * the rate of return and cancellation for each sale (%cancel_reject)

# <codecell>

data['%cancel_reject'] = 1-data['net_sale_count']/(data['net_sale_count']+data['rejected_returned_sale_count']+data['canceled_sale_count'])

# <markdowncell>

# We can then visualize some interesting facts. For example here are the 10 most returned/cancelled colors (the average being the red line):

# <codecell>

rejection_per_color = 1-(data[['colors', 'net_sale_count']].groupby('colors').sum().sum(axis=1)/data[['colors', 'net_sale_count', 'rejected_returned_sale_count', 'canceled_sale_count']].groupby('colors').sum().sum(axis=1))
r = rejection_per_color.order(ascending=False).head(10)
plt.barh(np.arange(len(r)), r.values, align='center', alpha=0.4)
plt.yticks(np.arange(len(r)), r.index)
plt.axvline(x=0.6, color='r')
plt.title('10 highests rates of rejection+cancellation per color')

# <markdowncell>

# Another way to improve the dataset is to use external data. In particular, weather data is useful in clothing retailing.
# We could add, for a given day, the temperature, precipitation level, or the comparison with normal seasonal levels. 

# <headingcell level=3>

# 2) With the given dataset, can you come up with a scientific approach and model for our ranking?

# <markdowncell>

# In order to order our observations, we need a score. Let's consider the following:

# <markdowncell>

# $$score_{item} = \log( \frac{\sum_{item}MIN(special\_price ; original\_price) * net\_sales}{\sum_{item} views\_count} )$$

# <markdowncell>

# Which is the net revenue per impression. We use the log in order to get more dispersion in the data.

# <markdowncell>

# We then calculate the score, along with the score on the last 7 and 30 days, plus the std deviation of those 3 scores:

# <codecell>

data['score'] = (data[['special_price','original_price']].min(axis=1)*data['net_sale_count']/data['impressions_count']).apply(log)

data['score_30'] = (data['special_price']*data['net_sale_count_last_30days']/data['impressions_count_last_30days'])
score_30_mean = data['score_30'][data['score_30'] < float('Inf')].mean()
data['score_30'] = data['score_30'].apply(lambda x : x if x < float('Inf') else score_30_mean).apply(log)

data['score_7'] = (data['special_price']*data['net_sale_count_last_7days']/data['impressions_count_last_7days'])
score_7_mean = data['score_7'][data['score_7'] < float('Inf')].mean()
data['score_7'] = data['score_7'].apply(lambda x : x if x < float('inf') else score_7_mean)
data['score_7'][data['score_7'] == 0] = data['score_30'][data['score_7'] == 0]
data['score_7'] = data['score_7'].apply(log)

data['score_std'] = data[['score','score_7','score_30']].std(axis=1);

# <markdowncell>

# The distribution of the score is the following:

# <codecell>

plt.hist(data['score'].values, bins = 10)
plt.title('Histogram of the score')

# <markdowncell>

# The distribution seems pretty close to a normal one.

# <markdowncell>

# Now let's try to model the score.
# First we need to turn our categorical variables ('brand'...) into column of 0/1 values:

# <codecell>

labels=[]
for col in qualitative:
    l = data[col].value_counts().order(ascending = False).index.tolist()[:-1]
    for label in l:
        data[label] = (data[col] == label)*1
        labels.append(label)

# <markdowncell>

# Now let's pick our features. 
# We will use all the previously created one, and quantitative data such as the price, %discount...

# <codecell>

features = labels + ['original_price', 'special_price', '%discount', 'day', 'score_7', 'score_30', 'score_std']
X = data[features]
mu, sigma = X.mean(), X.std()
X = (X-mu)/sigma # normalization of the data
y = data['score']

# <headingcell level=3>

# 3) How would you test, train, and evaluate your model?

# <markdowncell>

# We split our dataset into a training set (80% of the data) and a test set (20%).

# <codecell>

test = sample(set(data.index), int(0.20*len(y)))
training = list(set(data.index)-set(test))

# <markdowncell>

# Now let's use those features and train a regression model using random forests:

# <codecell>

clf = RandomForestRegressor()
clf = clf.fit(X.loc[training], y.loc[training])

# <markdowncell>

# Let's try to predict the scores on the test set:

# <codecell>

data['predictionsRF'].loc[test] = clf.predict(X.loc[test]);

# <markdowncell>

# Now, let's compare the ranking we got on the test set ('rank_model') set with the actual one ('rank_real'):

# <codecell>

data['rank_real'].loc[test] = data['score'].loc[test].order().rank()
data['rank_model'].loc[test] = data['predictionsRF'].loc[test].order().rank()
data[['rank_model','predictionsRF', 'rank_real','score']].loc[test][data['rank_model'].loc[test]<=10].sort('rank_model')

# <markdowncell>

# We can see that the model is performing reasonnably well on the test set.

# <markdowncell>

# Now let's imagine that we would like to recommend 10 items to a Female customer looking for a black dress 
# during summer. The recommendations would ve the following:

# <codecell>

user = X[(X['black'] > 0 ) & (X['Female'] > 0) & (X['Dresses'] > 0) & (X['Spring / Summer'] > 0)]
recommendations = DataFrame(clf.predict(user), index = user.index)[0]
recommendations = recommendations.order(ascending = False).head(10).index
data[qualitative.tolist() + ['score', 'predictionsRF']].loc[recommendations]

# <markdowncell>

# Which seeems like a reasonnable choice too.

