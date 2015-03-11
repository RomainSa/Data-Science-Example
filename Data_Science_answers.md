
# Introduction & Objective 

Zalora is a online fashion retailer. We have hundreds of thousands of products
accross all of our ventures -
Singapore, Indonesia, Malaysia, Hong Kong, Thailand, Philippines, Vietnam,
Brunei, Australia and New Zealand.
One of our challenging problems, is to determine which products are better than
others, essentially a venturewide
ranking for all of its products. The ranking can be different for each different
product category / subcategory.
The result of this ranking serves a lot of purposes:
Displaying best selling products on our homepage.
Sorting our catalogs / search results whenever users surf our homepage.
Sending better, more profitable email campaigns.

# Dataset

(Attached as a single products.csv file) Luckily, our Data Engineering team has
done a lot of data cleaning and
aggregation to produce everything in a clearly formatted table. The given data
set contains 4000 Indonesia
products and their metadatas that are currently available on our store, sorted
by a random order.
[Notice: a large part of this was randomized since it contains confidential
information]

# Questions

### 1) Give us your suggestions on how we could make our data set better / more useful.

First, let's import some packages, load the dataset and perform some treatments
on the date:

```python
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


    dataFile = '/home/roms/Desktop/Zalora/data-science-zalora/products.csv'
    data = DataFrame.from_csv(dataFile, encoding = 'ISO-8859-1')
    data['activated_at'] = data['activated_at'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    quantitative = data.select_dtypes(include=['int64', 'float64']).columns
    qualitative = data.select_dtypes(exclude=['int64', 'float64','datetime']).columns
    other_date = data.select_dtypes(include=['datetime']).columns
```
We have a dataset with:
* 4000 observations
* 25 columns, of which:
    * 6 are qualitative (ex: 'brand')
    * 18 are quantitative (ex: 'original_price')
    * 1 is a timestamp
* no missing data (but field 'special_price' is left empty when there is no
discount)

One thing we can do in order to improve the dataset is to calculate some
indicators, such as:
* the day of the year (1 being 1st of january) rather than a timestamp

```python
    data['day'] = data['activated_at'].apply(lambda x : x.timetuple().tm_yday)
```
* the number of views per impression for each item (%viewed)

```python
    data['%viewed'] = data['views_count']/data['impressions_count']
```
* the net number of items bought per view for each item (%bought)

```python
    data['%bought'] = (data['net_sale_count']/data['views_count'])
```
* the discount rate for each item (%discount)

```python
    data['special_price'] = data[['special_price','original_price']].min(axis=1)
    data['%discount'] = (1-data['special_price']/data['original_price'])
```
* the rate of return and cancellation for each sale (%cancel_reject)

```python
    data['%cancel_reject'] = 1-data['net_sale_count']/(data['net_sale_count']+data['rejected_returned_sale_count']+data['canceled_sale_count'])
```
We can then visualize some interesting facts. For example here are the 10 most
returned/cancelled colors (the average being the red line):

```python
    rejection_per_color = 1-(data[['colors', 'net_sale_count']].groupby('colors').sum().sum(axis=1)/data[['colors', 'net_sale_count', 'rejected_returned_sale_count', 'canceled_sale_count']].groupby('colors').sum().sum(axis=1))
    r = rejection_per_color.order(ascending=False).head(10)
    plt.barh(np.arange(len(r)), r.values, align='center', alpha=0.4)
    plt.yticks(np.arange(len(r)), r.index)
    plt.axvline(x=0.6, color='r')
    plt.title('10 highests rates of rejection+cancellation per color')
```

![png](data_science_example_files/data_science_example_21_1.png)


Another way to improve the dataset is to use external data. In particular,
weather data is useful in clothing retailing.
We could add, for a given day, the temperature, precipitation level, or the
comparison with normal seasonal levels.

### 2) With the given dataset, can you come up with a scientific approach and model for our ranking?

In order to order our observations, we need a score. Let's consider the
following:

$$score_{item} = \log( \frac{\sum_{item}MIN(special\_price ; original\_price) *
net\_sales}{\sum_{item} views\_count} )$$

Which is the net revenue per impression. We use the log in order to get more
dispersion in the data.

We then calculate the score, along with the score on the last 7 and 30 days,
plus the std deviation of those 3 scores:

```python
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
```
The distribution of the score is the following:

```python
    plt.hist(data['score'].values, bins = 10)
    plt.title('Histogram of the score')
```

![png](data_science_example_files/data_science_example_30_1.png)


The distribution seems pretty close to a normal one.

Now let's try to model the score.
First we need to turn our categorical variables ('brand'...) into column of 0/1
values:

```python
    labels=[]
    for col in qualitative:
        l = data[col].value_counts().order(ascending = False).index.tolist()[:-1]
        for label in l:
            data[label] = (data[col] == label)*1
            labels.append(label)
```
Now let's pick our features.
We will use all the previously created one, and quantitative data such as the
price, %discount...

```python
    features = labels + ['original_price', 'special_price', '%discount', 'day', 'score_7', 'score_30', 'score_std']
    X = data[features]
    mu, sigma = X.mean(), X.std()
    X = (X-mu)/sigma # normalization of the data
    y = data['score']
```
### 3) How would you test, train, and evaluate your model?

We split our dataset into a training set (80% of the data) and a test set (20%).

```python
    test = sample(set(data.index), int(0.20*len(y)))
    training = list(set(data.index)-set(test))
```
Now let's use those features and train a regression model using random forests:

```python
    clf = RandomForestRegressor()
    clf = clf.fit(X.loc[training], y.loc[training])
```
Let's try to predict the scores on the test set:

```python
    data['predictionsRF'] = clf.predict(X);
```
Now, let's compare the ranking we got on the test set ('rank_model') set with
the actual one ('rank_real'):

```python
    data['rank_real'] = data['score']
    data['rank_model'] = data['predictionsRF']
    data['rank_real'].loc[test] = data['score'].loc[test].rank(ascending = False)
    data['rank_model'].loc[test] = data['predictionsRF'].loc[test].rank(ascending = False)
    data[['rank_model','predictionsRF', 'rank_real','score']].loc[test][data['rank_model'].loc[test]<=10].sort('rank_model')
```



<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank_model</th>
      <th>predictionsRF</th>
      <th>rank_real</th>
      <th>score</th>
    </tr>
    <tr>
      <th>product_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>879239</th>
      <td>  1</td>
      <td> 8.471613</td>
      <td>  2</td>
      <td> 10.152595</td>
    </tr>
    <tr>
      <th>889036</th>
      <td>  2</td>
      <td> 8.435106</td>
      <td>  1</td>
      <td> 10.916467</td>
    </tr>
    <tr>
      <th>883028</th>
      <td>  3</td>
      <td> 8.223858</td>
      <td> 32</td>
      <td>  7.971915</td>
    </tr>
    <tr>
      <th>651925</th>
      <td>  4</td>
      <td> 8.156814</td>
      <td> 21</td>
      <td>  8.144761</td>
    </tr>
    <tr>
      <th>881856</th>
      <td>  5</td>
      <td> 8.130035</td>
      <td>  4</td>
      <td>  8.863287</td>
    </tr>
    <tr>
      <th>884123</th>
      <td>  6</td>
      <td> 8.109877</td>
      <td> 11</td>
      <td>  8.385713</td>
    </tr>
    <tr>
      <th>643491</th>
      <td>  7</td>
      <td> 8.098532</td>
      <td>  5</td>
      <td>  8.578641</td>
    </tr>
    <tr>
      <th>651439</th>
      <td>  8</td>
      <td> 8.063161</td>
      <td> 10</td>
      <td>  8.475313</td>
    </tr>
    <tr>
      <th>879284</th>
      <td>  9</td>
      <td> 8.027332</td>
      <td> 14</td>
      <td>  8.314014</td>
    </tr>
    <tr>
      <th>642347</th>
      <td> 10</td>
      <td> 7.985213</td>
      <td> 26</td>
      <td>  8.045734</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the model is performing reasonnably well on the test set.

Now let's imagine that we would like to recommend 10 items to a Female customer
looking for a black dress
during summer. The recommendations would ve the following:

```python
    user = X[(X['black'] > 0 ) & (X['Female'] > 0) & (X['Dresses'] > 0) & (X['Spring / Summer'] > 0)]
    recommendations = DataFrame(clf.predict(user), index = user.index)[0]
    recommendations = recommendations.order(ascending = False).head(10).index
    data[qualitative.tolist() + ['score', 'predictionsRF']].loc[recommendations]
```



<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colors</th>
      <th>gender</th>
      <th>season_group</th>
      <th>brand</th>
      <th>sub_cat_type</th>
      <th>cat_type</th>
      <th>score</th>
      <th>predictionsRF</th>
    </tr>
    <tr>
      <th>product_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>879236</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>          Lola Skye</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.253794</td>
      <td> 7.557819</td>
    </tr>
    <tr>
      <th>875224</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>             ZALORA</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.469794</td>
      <td> 7.384612</td>
    </tr>
    <tr>
      <th>642325</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td> Something Borrowed</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.383134</td>
      <td> 7.376071</td>
    </tr>
    <tr>
      <th>875221</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>             ZALORA</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.276299</td>
      <td> 7.246061</td>
    </tr>
    <tr>
      <th>270376</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>               Mint</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.097857</td>
      <td> 7.115752</td>
    </tr>
    <tr>
      <th>274397</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>              Mango</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.141375</td>
      <td> 6.987426</td>
    </tr>
    <tr>
      <th>283079</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>          Dei Reich</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 6.815274</td>
      <td> 6.975325</td>
    </tr>
    <tr>
      <th>642295</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td> Something Borrowed</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 6.998081</td>
      <td> 6.937949</td>
    </tr>
    <tr>
      <th>375830</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>     Noir Sur Blanc</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 7.065513</td>
      <td> 6.929591</td>
    </tr>
    <tr>
      <th>540035</th>
      <td> black</td>
      <td> Female</td>
      <td> Spring / Summer</td>
      <td>     EZRA by ZALORA</td>
      <td> Dresses</td>
      <td> Apparel and Accessories</td>
      <td> 6.875475</td>
      <td> 6.861785</td>
    </tr>
  </tbody>
</table>
</div>



Which seeems like a reasonnable choice too.
