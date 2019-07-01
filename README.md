# ML_futures

This is a machine learning project on predicting the commodity futures price.

Use as much information available to generate features and use these features to predict commodity futures price.
Available information includes:
  Date
  Option Code: each commodity futures has its own Option Code
  Close Price
  Open Price
  Inventory
  Warehouse Receipt
data are all available in raw. data is retrieved from 2007 to 2019. All data are retrieved from Wind.

Features:
  Basis Momentum: culmulative return of spot - culmulative returne of futures
  Bias Response Line: (price - moving average)/moving average * 100
  Inventory Percentage Change
  Return Signal Momentum: probability of positive return over the past r days
  Roll Return: (ln(spot) - ln(futures))*365/(maturity date - today)
  Open to Open return
  Close to Close return
  Warehouse Receipt Precentage Change
  

Strategy:
  In this project, I would use machine learning model to predict 37 commodity futures price with features mentioned above.
  
  Reasoning for data engineering: 
  I assume that there is some relationship between the features and the price movement. 
  However, Raw data is not enough to predict price movement. I assume that relationship of trend or reversal of price with a 
  periodic features is existed. Therefore, All features are calculated in several period, Eg, 5 days, 10 days. 
  Label is defined as two classes. If price has increased, marked as positive, else, marked as negativel.
  
  Reasoning for Machine learning process: 
  Use one year data as training set, 20 days data as test set. The prediction is rolling, 
  using a non-enchore walk forward method to predict futures price.
  The reason for walk forward method is that the relationship between features and label may be changed in different timeframe.
  Optimize and train the model in every step of the walk forward to ensure high quality of prediction. Cross validation is included
  in the optimization and training process to prevent overfitting.
  
  Currently, only xgboost is used as leaning machine. It is becauseXgboost usually has high performance in classification task. 
 
Vision Board:
  More data engineering to increase relationship between features and label (clipping, better outlier detection, pca)
  Include more learning machine, including not tree based machine.
  Include deep learning technique to ensemble result of several machines.
  Run time enhancement. (using multithreading, cuda or cython)
  Visualize predicting power
  Learn from the tree and data to conclude a certain relationship between features and label, for later direct usage
