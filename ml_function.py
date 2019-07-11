import pandas as pd
import numpy as np
import math
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, ParameterGrid

#initialize function
def initialize():
    '''
    Initialize all the neccesary variables and read the necessary data.
    
    Return
    ----------
    pd.Series: codelist (all the code of commoditiy futures), date
    null pd.DataFrame: output, true_and_pred
    null list: period_list, accuracy, precision, recall, f1, f2
    '''
    
    codelist = pd.read_csv(r"D:\QuantChina\product_name.csv", usecols= ["code"])
    date = pd.read_csv(r"D:\QuantChina\ML\rtn_data\all_rtn_data.csv", usecols = [0])

    output = pd.DataFrame()
    true_and_pred = pd.DataFrame()
    period_list = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    r2 = []

    output["date"] = date.iloc[:,0]
    true_and_pred["date"] = date.iloc[:,0]
    
    return codelist, output, true_and_pred, period_list, accuracy, precision, recall, f1, r2
    
#extracting data for x and y
def extract_data(codelist, k, regression, clipping, clip_benchmark):
    '''
    Extract all the features and label and reshaping them into correct way.
    label y is the next day label of today features
    
    Parameters
    ---------------
    regression: boolean
        if True, label is continuous return. Else, label is binary return, indicating positive or negative return
    clipping: boolean
        if True, label is clipped.
    clip_benchmark: float
        if clipping == True, 
        mark label as positive if y is greater than the percentile of clip_benchmark
        mark label as negative if y is less than the percentile of (1 - clip_benchmark)
        
    Return
    ---------------
    pd.DataFrame
        the first to the second last columns are features, the last column is label
        no None or nan data included, all data is valid
    '''
    product_name = str(codelist.iloc[k,0])
        
    X = []
    dtypelist = []
    
    data0 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\close_price_bw_pct_change_preprocessed.csv")
    #5,10,15,20,25,30,35,40,45,50
    X.append(data0)
    dtypelist.append("rtn_cc")

    data1 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\open_price_bw_pct_change_preprocessed.csv")
    X.append(data1)
    dtypelist.append("rtn_oo")
    #5,10,15,20,25,30,35,40,45,50

    data4 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\return_signal_momentum_preprocessed.csv")
    X.append(data4)
    dtypelist.append("rs")
    #5,10,15,20,25,30,35,40,45,50

    data6 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\roll_rtn_preprocessed.csv")
    X.append(data6)
    dtypelist.append("rr")
    #1

    data7 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\basis_momentum_preprocessed.csv")
    X.append(data7)
    dtypelist.append("bm")
    #5,10,15,20,25,30,35,40,45,50

    data11 = pd.read_csv(r"D:\QuantChina\ML\signal_data\inventory_bw_pct_change_preprocessed.csv")
    X.append(data11)
    dtypelist.append("inv_s")
    #5,10,15,20,25,30,35,40,45,50

    data12 = pd.read_csv(r"D:\QuantChina\ML\signal_data\warehouse_receipt_bw_pct_change_preprocessed.csv")
    X.append(data12)
    dtypelist.append("wr_s")
    #5,10,15,20,25,30,35,40,45,50

    data13 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\rsi_preprocessed.csv")
    X.append(data13)
    dtypelist.append("rsi")
    #5,10,15,20,25,30,35,40,45,50

    data14 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\bias_preprocessed.csv")
    X.append(data14)
    dtypelist.append("bias")
    #5,10,15,20,25,30,35,40,45,50
    
    data15 = pd.read_csv(r"D:\QuantChina\ML\signal_data\inventory_seasonality_preprocessed.csv")
    X.append(data15)
    dtypelist.append("sea_inv")

    data16 = pd.read_csv(r"D:\QuantChina\ML\signal_data\warehouse_receipt_seasonality_preprocessed.csv")
    X.append(data16)
    dtypelist.append("sea_wr")

    data17 = pd.read_csv(r"D:\QuantChina\ML\unadjusted\roll_rtn_seasonality_preprocessed.csv")
    X.append(data17)
    dtypelist.append("sea_rr")

    
    if regression == True or clipping == True:
        ydata = pd.read_csv(r"D:\QuantChina\ML\rtn_data\rtn_cc_data.csv")
    else:
        Y = pd.read_csv(r"D:\QuantChina\ML\unadjusted\open_price_fw_pct_change_5_signal.csv", usecols = [product_name])  
        test = pd.read_csv(r"D:\QuantChina\ML\unadjusted\open_price_fw_pct_change.csv", usecols = [product_name])
    
    X_df = pd.DataFrame()
    na = []

    for abc in range(len(X)):
        element = X[abc]
        a = element.columns.tolist()
        indexlist = [ele if ele[:3] == product_name for ele in a]
        
        #avoid null input
        if len(indexlist) == 0:
            continue
        
        na.append(tmp_tin.isnull().max())
        tmp_in = element.iloc[:,indexlist]
        tmp_in.columns += dtypelist[abc]
        X_df = pd.concat([X_df,tmp_in],axis=1)

    """
    #start training when there are 4 features available
    na.sort()
    na_count = na[4]
    """
    na_count = max(na)
    #use yesterday X to predict today Y
    X_df = X_df.iloc[na_count:-1].reset_index(drop=True)

    Y = Y.iloc[na_count+1:].reset_index(drop=True)
    
    data = pd.concat([X_df,Y], axis=1)
    data.dropna(inplace=True)
    
    if clipping == True:
        mean = data.iloc[:,-1].mean()
        std = data.iloc[:,-1].std()
        values_within = ((data.iloc[:,1] < mean + clip_benchmark*std) & (data.iloc[:,1] > mean - clip_benchmark*std))
        data.loc[values_within,product_name] = None
    
    return data, test

#calculate the daily return
def calc_rtn(product_name, tmp_true_and_pred, k, opt_marker, rtn_period):
    '''
    Calculate the daily return and culmulative return of individual product
    
    Parameters
    ----------------
    product_name: string
        name of that product
    tmp_true_and_pred: pd.DataFrame
        true and prediction
    k: int
        the no. of product that is using
    opt_marker: pd.DataFrame
        marking when to optimize
    rtn_period: int
        position period
        
    return
    ----------------
    pd.DataFrame
        1 col: true
        2 col: pred
        3 col: trn_olc
        4 col: rtn_co
        5 col: rtn_cc
        6 col: daily_rtn
        7 col: cum_rtn
    '''    
    #if opt_marker = 1, act according to prediction, if 0, use last prediction
    #make periodic position
    tmp_opt_marker = pd.Series(opt_marker.tolist())
    for abcdefg in range(tmp_opt_marker.shape[0]):
        if tmp_opt_marker.iloc[abcdefg] == 1:
            if abcdefg + rtn_period < tmp_opt_marker.shape[0]:
                tmp_opt_marker.iloc[abcdefg + rtn_period] = 1


    tmp_tmp_true_and_pred = []
    difference = tmp_opt_marker.shape[0] - tmp_true_and_pred.shape[0]
    for abcdefg in range(tmp_opt_marker.shape[0]):
        if abcdefg < difference:
            continue
        if tmp_opt_marker.iloc[abcdefg] == 1:
            for cba in range(rtn_period):
                if len(tmp_tmp_true_and_pred) + 1 <= tmp_true_and_pred.shape[0] and abcdefg-difference+cba < tmp_true_and_pred.shape[0]:
                    if str(tmp_true_and_pred.iloc[abcdefg-difference + cba, 0]) != "nan":
                        tmp_tmp_true_and_pred.append(tmp_true_and_pred.iloc[abcdefg - difference,1])
                    else:
                        tmp_tmp_true_and_pred.append(None)
    
    '''
    if len(tmp_tmp_true_and_pred) < tmp_true_and_pred.shape[0]:
        tmp_true_and_pred = tmp_true_and_pred.iloc[tmp_true_and_pred.shape[0] - len(tmp_tmp_true_and_pred):]
    '''
    
    #read return data
    rtn_data = pd.read_csv(r"D:\QuantChina\ML\rtn_data\all_rtn_data.csv",usecols = [k*3+1,k*3+2,k*3+3])
    
    tmp_output = pd.DataFrame()
    tmp_output[product_name + "_true_value"] = tmp_true_and_pred.iloc[:,0]
    tmp_output[product_name + "_position"] = tmp_tmp_true_and_pred
    tmp_output = tmp_output.reset_index(drop=True)

    rtn_data = rtn_data.iloc[rtn_data.shape[0] - tmp_true_and_pred.shape[0]:].reset_index(drop=True)
    order = [product_name + "_rtn_olc", product_name + "_rtn_co", product_name+"_rtn_cc"]
    rtn_data = rtn_data[order]
    tmp_output = pd.concat([tmp_output,rtn_data], axis=1)
    
    daily_rtn = []
    
    flag = 0
    for i in range(tmp_output.shape[0]):
        if str(tmp_output[product_name + "_position"].iloc[i]) == "nan" and flag == 0:
            daily_rtn.append(None)
        elif str(tmp_output[product_name + "_position"].iloc[i]) == "nan" and flag == 1:
            daily_rtn.append(0)

        elif i == 0:
            if tmp_output[product_name + "_position"].iloc[i] == 1:
                daily_rtn.append(tmp_output[product_name + "_rtn_co"].iloc[i])
            elif tmp_output[product_name + "_position"].iloc[i] == 0:
                daily_rtn.append(0.0)
            else:
                daily_rtn.append(-tmp_output[product_name + "_rtn_co"].iloc[i]) 
        elif tmp_output[product_name + "_position"].iloc[i] == 1:
            flag = 1
            if tmp_output[product_name + "_position"].iloc[i-1] == 1:
                daily_rtn.append(tmp_output[product_name + "_rtn_cc"].iloc[i])
            elif tmp_output[product_name + "_position"].iloc[i-1] == 0:
                daily_rtn.append(tmp_output[product_name + "_rtn_co"].iloc[i])
            else:
                daily_rtn.append((1-tmp_output[product_name + "_rtn_olc"].iloc[i])*(1+tmp_output[product_name + "_rtn_co"].iloc[i])-1)
        elif tmp_output[product_name + "_position"].iloc[i] == 0:
            flag = 1
            if tmp_output[product_name + "_position"].iloc[i-1] == 1:
                daily_rtn.append(tmp_output[product_name + "_rtn_olc"].iloc[i])
            elif tmp_output[product_name + "_position"].iloc[i-1] == 0:
                daily_rtn.append(0.0)
            else:
                daily_rtn.append(-tmp_output[product_name + "_rtn_olc"].iloc[i])
        else:
            flag = 1
            if tmp_output[product_name + "_position"].iloc[i-1] == 1:
                daily_rtn.append((1+tmp_output[product_name + "_rtn_olc"].iloc[i])*(1-tmp_output[product_name + "_rtn_co"].iloc[i])-1)
            elif tmp_output[product_name + "_position"].iloc[i-1] == 0:
                daily_rtn.append(-tmp_output[product_name + "_rtn_co"].iloc[i])
            else:
                daily_rtn.append(-tmp_output[product_name + "_rtn_cc"].iloc[i])
    tmp_output[product_name + "_daily_rtn"] = daily_rtn
                
    cum_rtn = []
    for i in range(tmp_output.shape[0]):
        if i == 0:
            cum_rtn.append(1.0)
            cum_rtn.append(cum_rtn[-1]*(1+daily_rtn[i]))
        else:
            cum_rtn.append(cum_rtn[-1]*(1+daily_rtn[i]))
    cum_rtn = cum_rtn[1:]
    tmp_output[product_name + "_cum_rtn"] = cum_rtn
    
    return tmp_output, tmp_opt_marker


#calculate the sumyield of the portfolio
def calc_sumyield(output, date, codelist):
    '''
    Calculate total rtn of portfolio and drawdown
    
    Parameters
    -------------
    output: pd.DataFrame
        all output from calc_rtn
    date: pd.DataFrame
    codelist: pd.DataFrame
    
    Return
    -------------
    pd.DataFrame
        original output concat with total cum rtn and drawdown
    pd.DataFrame
        daily rtn for all individual product and portfolio
    pd.DataFrame
        cum rtn for all individual product and portfolio
    '''
    rtn = pd.DataFrame()
    tmp_rtn = pd.DataFrame()

    tmp_rtn["date"] = date
    for a in range(codelist.shape[0]):
        product_name = str(codelist.iloc[a,0])
        try:
            rtn[product_name] = output[product_name + "_daily_rtn"]
        except:
            continue

    tmp_rtn = pd.concat([tmp_rtn,rtn], axis=1)
    t_daily_rtn = []
    t_cum_rtn = []
    t_drawdown = []
    t_cum_rtn.append(1.0)
    maxi = 1.0

    flag = 0
    for i in range(rtn.shape[0]):
        count = 0
        tdy_rtn = rtn.iloc[i]
        for j in range(rtn.shape[1]):
            if str(tdy_rtn.iloc[j]) != "nan":
                count += 1
        t_daily_rtn.append(tdy_rtn.sum()/count)

        if flag == 1:
            t_cum_rtn.append(t_cum_rtn[-1]*(1+t_daily_rtn[-1]))
        elif str(t_daily_rtn[-1]) == "nan":
            t_cum_rtn.append(np.nan)
        else:
            t_cum_rtn.append(1*(1+t_daily_rtn[-1]))
            flag = 1

        if t_cum_rtn[-1] > maxi:
            maxi = t_cum_rtn[-1]

        try:
            t_drawdown.append(t_cum_rtn[-1]/maxi - 1)
        except:
            t_drawdown.append(None)

    t_cum_rtn = t_cum_rtn[1:]

    output["daily_rtn"] = t_daily_rtn
    output["sumyield"] = t_cum_rtn
    output["drawdown"] = t_drawdown

    cumrtn = pd.DataFrame()

    cumrtn["date"] = date

    for a in range(codelist.shape[0]):
        product_name = str(codelist.iloc[a,0])
        try:
            cumrtn[product_name] = output[product_name + "_cum_rtn"]
        except:
            continue

    tmp_rtn["sumyield"] = output["daily_rtn"]
    cumrtn["sumyield"] = output["sumyield"]

    return output, tmp_rtn, cumrtn

#calculate the metrics of the portfolio
def calc_metrics(rtn,cumrtn, accuracy, r2, regression, date, evals):
    '''
    Calculate cum_rtn, annualized_rtn, annualized_volatility, sharpe, and max drawdown for all individual product and portfolio
    
    Parameters
    --------------
    rtn: pd.DataFrame
        daily rtn
    cumrtn: pd.DataFrame
    accuracy: list
    r2: list
    regression: boolean
    date: pd.Series
    
    return
    -------------
    pd.DataFrame
    '''
    metrics = pd.DataFrame()
    tmp_cumrtn = cumrtn.iloc[:,1:]

    cum_rtn = []
    annualized_rtn = []
    annualized_volatility = []
    sharpe = []
    max_drawdown = []

    for j in range(tmp_cumrtn.shape[1]):
        cum_rtn.append(tmp_cumrtn.iloc[-1,j])
        
        first_date = datetime.datetime.strptime(date.iloc[tmp_cumrtn.iloc[:,j].first_valid_index()],"%m/%d/%Y")
        last_date = datetime.datetime.strptime(date.iloc[-1],"%m/%d/%Y")
        delta = (last_date - first_date).days
        
        annualized_rtn.append((np.exp(np.log(tmp_cumrtn.iloc[-1,j])*365/delta) - 1))
        annualized_volatility.append(np.std(rtn.iloc[:,j+1])* math.sqrt(252))
        sharpe.append(annualized_rtn[-1]/annualized_volatility[-1])

        maxi = 1.0
        tmp = []
        for i in range(tmp_cumrtn.iloc[:,j].isnull().sum()):
            tmp.append(np.nan)
        for i in range(tmp_cumrtn.iloc[:,j].isnull().sum(),tmp_cumrtn.shape[0]):
            if tmp_cumrtn.iloc[i,j] > maxi:
                maxi = tmp_cumrtn.iloc[i,j]
            tmp.append(tmp_cumrtn.iloc[i,j]/maxi-1)
        max_drawdown.append(np.nanmin(tmp))

    metrics["cumulative_rtn"] = cum_rtn
    metrics["annualized_rtn"] = annualized_rtn
    metrics["annualized_volatility"] = annualized_volatility
    metrics["sharpe"] = sharpe
    metrics["max_drawdown"] = max_drawdown

    if regression == True:
        r2.append(None)
        metrics["r2"] = r2
    else:
        accuracy.append(None)
        metrics["accuracy"] = accuracy
        evals.append(None)
        metrics["evals"] = evals
    metrics = metrics.set_index(cumrtn.columns[1:])
    
    return metrics

#binary classification of prediction
def classify(y_pred, benchmark, regression):
    if regression == True:
        y_pred = [1 if y >= benchmark else -1 if y < -benchmark else 0 for y in y_pred]
    else:
        y_pred = [1 if y >= benchmark else -1 if y < benchmark else 0 for y in y_pred]
    return y_pred

#walk forward generator
def wf(data, n_samples, test_percentage, test_num, isper):
    '''
    generate walk foward data
    
    Parameters
    ---------------
    data: pd.DataFrame
    n_samples: int
        number of training set
    test_percentage: float
        percentage of test set
    test_num: int
        number of test set
    iseper: boolean
        if True, use test_percentage. Else, use test_num
        
    Return
    ---------------
    int
        total length of valid data
    int
        length of test set
    int
        how many steps the walk forward process is included
    '''
    if isper:
        length = data.shape[0] - data.iloc[:,0].isna().sum()
        test_length = int(round(n_samples*test_percentage))
        steps = int(round((length - n_samples)/test_length,0))
        return length, test_length, steps
    else:
        length = data.shape[0] - data.iloc[:,0].isna().sum()
        test_length = test_num
        steps = int(round((length - n_samples)/test_length,0))
        return length, test_length, steps

def build_model(data_wf, test_percentage, test_num, isper, method, params, opt_and_train, benchmark, regression, rtn_period):
    '''
    learning machine
    
    Parameters
    ----------------
    data_wf: pd.DataFrame
    test_percentage: float
    test_num: int
    isper: boolean
    method: string
        "xgboost" or "random_forest"
    params: dictionary
        hyper-parameters for learning machine
    opt_and_train: boolean
        if True, optimize hyper-parameters when train. Else, no optimization
    benchmark: float
        benchmark for binary classification. [0:1]
    regression: boolean
    
    Return
    ----------------
    pd.DataFrame
        1 col: true
        2 col: pred
    object
        learning model
    '''
    y = data_wf.iloc[:,-1]
    X = data_wf.iloc[:,:-1]  
    
    #test train split
    if isper:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_percentage, shuffle = False)
        
    else:
        if rtn_period > 1:
            X_train = X.iloc[:-test_num]
            X_test = X.iloc[-test_num:]
            y_train = y.iloc[:-test_num]
            y_test = y.iloc[-test_num:]
        else:
            X_train = X.iloc[:-test_num - rtn_period]
            X_test = X.iloc[-test_num:]
            y_train = y.iloc[:-test_num - rtn_period]
            y_test = y.iloc[-test_num:]
        
    #optimization and training at the same date
    if opt_and_train == 1:    
        results = pd.DataFrame()
        hyperparams = {"max_depth": [6,7,8]}
        
        #xgb.cv method
        dtrain = xgb.DMatrix(X_train,label=y_train)
        summary_list = []
        params = {'min_child_weight': 0.5, "gamma":0,
                  'colsample_bytree': 0.5, 'subsample': 0.3, 
                  "n_jobs": -1, "verbosity": 0, 
                  "silent": 1, "eta":0.1,
                  "n_estimators": 900}
        for hyperparams in ParameterGrid(hyperparams):

            params.update(hyperparams)
            summary = xgb.cv(params, dtrain, num_boost_round=4096, nfold=3, metrics = "error", early_stopping_rounds = 50,shuffle = False)
            
            best_cv_score = min(summary.iloc[:,0])
            params["n_estimators"] = np.argmin(summary.iloc[:,0])            
            results = results.append({"Score": best_cv_score,
                                      "Parameters": params}, ignore_index=True)
            
        params = results["Parameters"].iloc[results["Score"].idxmin()]
        model = xgb.XGBClassifier(**params)    
       
        
        #gridsearch cv
        '''
        #n_jobs = no. of precessors, -1 to using all precessors
        if regression == True:
            params = {'min_child_weight': 0.5, "gamma":0,
                      'colsample_bytree': 0.5, 'subsample': 0.3, 
                      "n_estimators": 900, "n_jobs": -1, 
                      "verbosity": 0, "silent": 1,
                      "eta":0.1}
            model = xgb.XGBRegressor(**params)
            search = GridSearchCV(model, hyperparams, n_jobs = -1, iid = False, scoring = "r2", cv=3)
        else:
            params = {"objective": "binary:logistic", 
                      'min_child_weight': 0.5, "gamma":0,
                      'colsample_bytree': 0.5, 'subsample': 0.3, 
                      "n_estimators": 900, "n_jobs": -1, 
                      "verbosity": 0, "silent": 1,
                      "eta":0.1, "max_depth": 10}
            model = xgb.XGBClassifier(**params)
            search = GridSearchCV(model, hyperparams, n_jobs = -1, iid = False, scoring = ["accuracy", "f1"], refit = "accuracy", cv=3)
        
        search.fit(X_train, y_train)
        model = search
        print model.best_params_
        #print model.get_params()["estimator__n_estimators"]
        '''
    else:
        model = xgb.XGBClassifier(**params)

        
    #ml process
    if method == "xgboost":
        #validation split
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)
        
        #model.fit(X_train,y_train, eval_set=[(X_val, y_val)], eval_metric = "error", early_stopping_rounds = 5, verbose = False)
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        confidence = model.predict_proba(X_test)

        if regression != True:
            pred = classify(y_pred, benchmark, regression)
        
        return y_test, pred, confidence
        
    else:
        if method == "random_forest":
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred = classify(y_pred, benchmark, regression)
        y_test = pd.DataFrame(y_test).reset_index()
        y_test[y.name + "_prediction"] = y_pred

        return y_test, model
