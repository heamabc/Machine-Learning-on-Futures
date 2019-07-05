import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

from ml_function import *
%load_ext autoreload
%autoreload 1
%aimport ml_function


if__name__== "__main__:
    #periodic prediction and optimization
    #train and optimize dataset at the same date for all product

    #general parameters
    isper = False
    method = "xgboost"
    test_percentage = 0.1
    encore = False
    regression = False
    clipping = False
    clip_benchmark = 0.0
    rtn_period = 5

    #opt params
    #if opt_and_train = True, test_num is the period for opt_and_train
    optimize = True
    opt_freq = 5
    opt_and_train = True

    #ML params
    if regression == True:
        params = {"eta":0.1, "objective": "reg:sqauredlogerror",
                  'n_estimators': 1000, 'max_depth': 4, 'min_child_weight': 0.5, "gamma":0,
                  'colsample_bytree': 0.2, 'subsample': 0.2, "n_jobs": -1, "verbosity": 0}
    else:
        params = {"eta":0.1, "objective": "binary:logistic",
                  'n_estimators': 1000, 'max_depth': 4, 'min_child_weight': 0.5, "gamma":0,
                  'colsample_bytree': 0.2, 'subsample': 0.2, "n_jobs": -1, "verbosity": 0}
    benchmark = 0.5
    train_size = 252

    #initialize variables
    codelist, output, true_and_pred, period_list, accuracy, precision, recall, f1, r2= initialize()


    #prevent error
    if opt_and_train == True:
        isper = False

    #ML enginedate
    for test_num in tqdm([20], desc="test_num"):

        n_samples = train_size + test_num
        for k in tqdm(range(codelist.shape[0]), desc="codelist"):        
            product_name = str(codelist.iloc[k,0])
            if product_name in ["319","320"]:
                continue
            data = extract_data(codelist, k, regression, clipping, clip_benchmark)
            length, test_length, steps = wf(data, n_samples, test_percentage, test_num, isper)

            #generate opt_marker
            if product_name == "305":
                tmplist = []

                if rtn_period > 1:
                    add = rtn_period
                else:
                    add = 0

                for abcdefg in range(output.shape[0] - (data.shape[0]+add) + train_size):
                    tmplist.append(None)

                for abcdefg in range(output.shape[0] - (output.shape[0] - (data.shape[0]+add) + train_size)):
                    if abcdefg % test_num == 0:
                        tmplist.append(1)
                    else:
                        tmplist.append(0)
                opt_marker = pd.Series(tmplist)

            tmp_true_and_pred = pd.DataFrame()

            #start ML
            na_count = output.shape[0] - (data.shape[0]+add)
            for i in tqdm(range(data.shape[0]+test_num), desc = "ML"):
                if opt_marker.iloc[i+na_count-test_num] == 1 and i >= n_samples:
                    if i <= data.shape[0]:
                        data_wf = data.iloc[i-n_samples:i]
                        y_test, model = build_model(data_wf, test_percentage, test_num, isper, method, params, opt_and_train, benchmark, regression, rtn_period)
                    else:
                        data_wf = data.iloc[i-n_samples:]
                        y_test, model = build_model(data_wf, test_percentage, data_wf.shape[0] - train_size, isper, method, params, opt_and_train, benchmark, regression, rtn_period)
                    tmp_true_and_pred = tmp_true_and_pred.append(y_test, ignore_index=True)
                else:
                    continue

            if rtn_period > 1:
                for abc in range(rtn_period):
                    tmp_true_and_pred = tmp_true_and_pred.append(pd.Series(), ignore_index = True)

            #reshaping to same length
            tmp_true_and_pred = tmp_true_and_pred.drop("index", axis=1)


            if regression == True:
                tmp_true_and_pred[product_name+"_signal"] = [1 if x>= benchmark else -1 if x < -benchmark else 0 for x in tmp_true_and_pred[product_name+"_prediction"]]
                order = []
                order.append(tmp_true_and_pred.columns[0])
                order.append(tmp_true_and_pred.columns[2])
                order.append(tmp_true_and_pred.columns[1])
                tmp_true_and_pred[order]
            else:
                tmp_true_and_pred[product_name] = [1 if x == 1 else -1 if x == 0 else 0 for x in tmp_true_and_pred[product_name]]

            '''
            if rtn_period > 1:
                for abc in range(rtn_period - 1):
                    tmp_true_and_pred = tmp_true_and_pred.append(pd.Series(), ignore_index=True)
            tmp_true_and_pred.reset_index(drop=True)
            '''
            if true_and_pred.shape[0] > tmp_true_and_pred.shape[0]:
                for abc in range(true_and_pred.shape[0] - tmp_true_and_pred.shape[0]):
                    tmp_true_and_pred.index+=1
            true_and_pred = pd.concat([true_and_pred,tmp_true_and_pred], axis=1)

            #calculate daily return
            tmp_output, tmp_opt_marker = calc_rtn(product_name, tmp_true_and_pred, k, opt_marker, rtn_period)

            if rtn_period == 1:
                if regression == True:
                    r2.append(r2_score(tmp_true_and_pred.iloc[:,0].dropna(), tmp_true_and_pred.iloc[:,2].dropna()))
                else:
                    accuracy.append(accuracy_score(tmp_true_and_pred.iloc[:,0].dropna(), tmp_true_and_pred.iloc[:,1].dropna()))
                    precision.append(precision_score(tmp_true_and_pred.iloc[:,0].dropna(), tmp_true_and_pred.iloc[:,1].dropna()))
                    recall.append(recall_score(tmp_true_and_pred.iloc[:,0].dropna(), tmp_true_and_pred.iloc[:,1].dropna()))
                    f1.append(f1_score(tmp_true_and_pred.iloc[:,0].dropna(), tmp_true_and_pred.iloc[:,1].dropna()))
            else:
                tmp_true = []
                tmp_pred = []
                diff = tmp_opt_marker.shape[0] - tmp_true_and_pred.shape[0]
                for abcdefg in range(opt_marker.shape[0]):
                    if abcdefg < tmp_opt_marker.shape[0] - tmp_true_and_pred.shape[0]:
                        continue
                    if tmp_opt_marker.iloc[abcdefg] == 1 and str(tmp_true_and_pred.iloc[abcdefg - diff,1]) != "nan":
                        tmp_true.append(tmp_true_and_pred.iloc[abcdefg - diff,0])
                        tmp_pred.append(tmp_true_and_pred.iloc[abcdefg - diff,1])

                accuracy.append(accuracy_score(tmp_true, tmp_pred))


            if tmp_output.shape[0] < output.shape[0]:
                for abcdefg in range(output.shape[0] - tmp_output.shape[0]):
                    tmp_output.index += 1

            output = pd.concat([output,tmp_output], axis=1)


        date = output["date"]
        #calculate the sumyield of the portfolio
        output, rtn, cumrtn = calc_sumyield(output, date, codelist)

        #metrics calculator
        metrics =  calc_metrics(rtn,cumrtn, accuracy, r2, regression)
  
    writer = pd.ExcelWriter(r'D:\QuantChina\ML\backtest_10_features_252+20_oo_xgbcv.xlsx', engine='xlsxwriter')
    output.to_excel(writer, sheet_name = "data", index=False)
    true_and_pred.to_excel(writer, sheet_name = "true_and_pred", index=False)
    rtn.to_excel(writer, sheet_name = "daily_rtn", index=False)
    cumrtn.to_excel(writer, sheet_name = "cumulative_rtn", index=False)
    metrics.to_excel(writer, sheet_name = "metrics")
    writer.save()
    
    return
