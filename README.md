# Machine Learning on Futures

This is a machine learning project on predicting the 37 Chinese commodity futures price. it uses Wind information to generate features and use these features to predict commodity futures price.

### Performance
Optained on average 0.57 accuracy. Achieved 10.7% annualized return, and sharpe ratio of 1.5.

### File Description
- features_generator.py and preprocessing.py : for generating features and preprocessing them
- ml_function.py : contains the neccessary function for running the program (read data, cal return, machine learning)
- main.py 

### Available information includes
- Date
- Option Code: each commodity futures has its own Option Code
- Spot Price
- Close Price: rolling month-to-month contracts
- Open Price: rolling month-to-month contracts
- Inventory
- Warehouse Receipt

data are all available in raw. data is retrieved from 2007 to 2019. All data are retrieved from Wind.

### Features
- Basis Momentum: culmulative return of spot - culmulative returne of futures : period = 5,10,15
- Bias Response Line: (price - moving average)/moving average * 100 : period = 5,10,15
- Return Signal Momentum: probability of positive return over the past r days : period = 5,10,15
- Roll Return: (ln(spot) - ln(futures))*365/(maturity date - today)
- Time Series Momentum : culmulative return 
- Inventory Percentage Change : period = 5,10,15
- Open to Open return : period = 5,10,15
- Close to Close return : period = 5,10,15
- Warehouse Receipt Precentage Change : period = 5,10,15
- Seasonality Features:
 - - Inventory year-to-year Percentage Change
 - - Warehouse Receipt year-to-year Percentage Change
 - - Roll Return year-to-year Percentage Change

# Strategy
In this project, I would use machine learning model to predict 37 commodity futures price with features mentioned above.

#### Operation and reasoning for data engineering: 
 I assume that there is some relationship between the features and the price movement. However, Raw data is not enough to predict price movement. I assume that relationship of trend or reversal of price with a periodic features is existed. Therefore, All features are calculated in several period, Eg, 5 days, 10 days. Label is defined as two classes. If price has increased, marked as positive, else, marked as negativel.

#### Operation and reasoning for Machine learning process: 
Use one year data as training set, 20 days data as test set. The prediction is rolling, For any given date, if there is 5 features data available, the machine learning process will be initiatedusing a non-enchore walk forward method to predict futures price. (ie. the training set will be fixed in length)The reason for walk forward method is that the relationship between features and label may be changed in different timeframe.Optimize and train the model in every step of the walk forward to ensure high quality of prediction. Cross validation is included to perform optimization.Only optimize the depth of tree. (From emperical testing, the best depth of tree fluctuates a lot.) Currently, only xgboost is used as leaning machine. It is becauseXgboost usually has high performance in classification task. 

## Vision Board
- More data engineering to increase relationship between features and label (clipping, better outlier detection, pca)
- Include more types of learning machine models
- Include deep learning technique to ensemble result of several machines.
- Run time enhancement. (using multithreading, cuda or cython)
- Visualize predicting power
- Learn from the tree and data to conclude a certain relationship between features and label, for later direct usage
