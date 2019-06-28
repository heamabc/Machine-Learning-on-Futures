import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

def pct_change(bw, data, periods):
    '''
    calculate backward percentage change

    Parameters
    ----------
    bw: boolean
      If True, calculate backward percentage change. Else, forward percentage change
    data : pd.DataFrame
      The first column must be date, all other columns are data to be calculated.
    periods: list
      list of periods of percentage change to be calculated

    Returns
    ----------
    pd.DataFrame
      first column is date, all other are percentage change
    ''' 
  
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    if bw == True:
        for j in range(data.shape[1]):
            for r in periods:
                output[data.iloc[:,j].name + "_" + str(r)] = data.iloc[:,j]/data.iloc[:,j].shift(r) - 1
    else:
        for j in range(data.shape[1]):
            for r in periods:
                tmpser = data.iloc[:,j]/data.iloc[:,j].shift(r) - 1
                tmpser = tmpser.iloc[r:].reset_index(drop=True)
                output[data.iloc[:,j].name + "_" + str(r)] = tmpser
    return output

def med_outlier(data):
    '''
    Detect outlier that exceeds median +/- 5*median1, replace with median +/- 5*median1,
    median1 = |x - median|.median
    '''
    output = pd.DataFrame()
    output["date"] = data["date"]
    data = data.iloc[:,1:]

    for j in range(data.shape[1]):
        median = np.median(data.iloc[:,j])
        median1 = np.median(abs(data.iloc[:,j] - np.mean(data.iloc[:,j])))

        f = lambda x: median + 5*median1 if x > median + 5*median1 else median - 5*median1 if x < median - 5*median1 else x
        output[data.iloc[:,j].name] = data.iloc[:,j].apply(f)
    return output
  
def standardize(data):
    #standardize data to N(0,1)
    output = pd.DataFrame()
    output["date"] = data["date"]
    name = data.columns
    data = data.iloc[:,1:]

    scaler = StandardScaler()
    output = pd.concat([output,pd.DataFrame(scaler.fit_transform(data))], axis=1)
    output.columns = name
    return output
  
def rank_engine(data):
    '''
    data reqruiement: same product with different period must be placed adjacently
    return rank df with rank of data with the same period
    '''
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    #count the number of products
    a = data.columns.tolist()
    for i in range(len(a)):
        a[i] = a[i][:3]
    a = pd.Series(a)
    length = a.nunique()

    for k in range(data.shape[1]/length):
        tmp_data = pd.DataFrame()
        for j in range((length)):
            tmp = data.iloc[:,j*(data.shape[1]/length) + k]
            tmp_data = pd.concat([tmp_data,tmp], axis=1)

        tmp_data = tmp_data.rank(axis=1,method="first", ascending=False)
        output = pd.concat([output,tmp_data], axis=1)
    return output

def binarize(data, zero_centered):
    output = pd.DataFrame()
    output['date'] = data.iloc[:,0]
    data = data.iloc[:,1:]

    if zero_centered == True:
        for j in range(data.shape[1]):
            output[data.iloc[:,j].name] = [1 if x >=0 else 0 if x < 0 else None for x in data.iloc[:,j]]
    else:
        for j in range(data.shape[1]):
            output[data.iloc[:,j].name] = [1 if x >=1 else 0 if x < 1 else None for x in data.iloc[:,j]]
    return output

def match_date(data, date):
    '''
    make all data matched date with the date column
    if today has no data, use last effective date data to replace today data

    Parameters
    ----------
    data: pd.DataFrame
      even column is the date, odd column is the corresponding data of that feature
    date: pd.Series
      date that would like to be matched

    Returns
    ----------
    pd.DataFrame
      first column is date, all other are matched date data
    '''
    output = pd.DataFrame()
    output["date"] = date.iloc[:,0]
    effective_date = ""
    effective_data = None

    for a in range(data.shape[1]/2):
        product_name = str(data.iloc[:,a*2].name[0:3])
        tmp_list = []
        flag = 0
        data_count = data.iloc[:,a*2].shape[0] - data.iloc[:,a*2].isnull().sum()

        for i in range(date.shape[0]):       

            today = datetime.datetime.strptime(date.iloc[i,0], "%m/%d/%Y")
            first_date = datetime.datetime.strptime(data.iloc[0,a*2], "%m/%d/%Y")

            #3 cases:
            #first: today is earlier than the first available data, than today data is none
            #second: today match with the date that data is available, use the first effective data of that date, if no effective data, use             last effective data
            #third: today does not match the date that data is available, use the first last effective data.
            if first_date > today:
                tmp_list.append(None)        
            elif '{d.month}/{d.day}/{d.year}'.format(d=today) in data.iloc[:,a*2].tolist():
                tmp_date = date.iloc[i,0]
                tmp_data = data[data[product_name + "_date"] == tmp_date]

                #if today has multiple data, use only the first useful data. If no useful data, effective date
                #and effective data remain unchanged
                for kk in range(tmp_data.shape[0]):
                    if str(tmp_data.iloc[kk,a*2+1]) != "nan" and str(tmp_data.iloc[kk,a*2+1]) != "None" and str(tmp_data.iloc[kk,a*2+1]) != "0" and tmp_data.iloc[kk,a*2+1] != 0:
                        effective_date = tmp_date
                        effective_data = tmp_data.iloc[kk,a*2+1]
                        break

                tmp_list.append(effective_data)
            else:
                delta_list = []

                #have effective date
                if flag != 0 and str(tmp_list[-1]) != "nan" and str(tmp_list[-1]) != "None":
                    index = data[data[product_name + "_date"] == effective_date].index[0]

                    for j in range(100):
                        if index + j == data_count:
                            break
                        tmp_date = datetime.datetime.strptime(data.iloc[index + j,a*2], "%m/%d/%Y")
                        delta = tmp_date - today
                        if delta.days > 0:
                            break

                    if str(data.iloc[index + j,a*2+1]) != "nan" and str(data.iloc[index + j,a*2+1]) != "None" and str(data.iloc[index + j,a*2+1]) != "0" and data.iloc[index+j,a*2+1] != 0:
                        delta_list.append(delta.days)

                    if len(delta_list) > 0:
                        for j in range(len(delta_list)):
                            if delta_list[j] == max(delta_list):

                                tmp_date = today + datetime.timedelta(delta_list[j])
                                tmp_date = '{d.month}/{d.day}/{d.year}'.format(d=tmp_date)
                                tmp_data = data[data[product_name + "_date"] == tmp_date]

                                #if today has multiple data, use only the first useful data. If no useful data, effective date
                                #and effective data remain unchanged
                                for kk in range(tmp_data.shape[0]):
                                    if str(tmp_data.iloc[kk,a*2+1]) != "nan" and str(tmp_data.iloc[kk,a*2+1]) != "None" and str(tmp_data.iloc[kk,a*2+1]) != "0" and tmp_data.iloc[kk,a*2+1] != 0:
                                        effective_date = tmp_date
                                        effective_data = data[data[product_name + "_date"] == effective_date].iloc[kk,a*2+1]
                                        break
                                break

                    tmp_list.append(effective_data)
                    flag = 1

                else:
                    #no effective date and search for effective date
                    for j in range(data_count):
                        #if data date is nan then break
                        if str(data.iloc[j,a*2]) == "nan":
                            break
                        tmp_date = datetime.datetime.strptime(data.iloc[j,a*2], "%m/%d/%Y")
                        delta = tmp_date - today
                        if delta.days>0:
                            break
                        if str(data.iloc[j, a*2+1]) != "nan" and str(data.iloc[j,a*2+1]) != "None" and data.iloc[j,a*2+1] != 0:
                            delta_list.append(delta.days)

                    if len(delta_list) > 0:
                        for j in range(len(delta_list)):
                            if delta_list[j] == max(delta_list):

                                tmp_date = today + datetime.timedelta(delta_list[j])
                                tmp_date = '{d.month}/{d.day}/{d.year}'.format(d=tmp_date)
                                tmp_data = data[data[product_name + "_date"] == tmp_date]

                                #if today has multiple data, use only the first useful data. If no useful data, effective date
                                #and effective data remain unchanged
                                for kk in range(tmp_data.shape[0]):
                                    if str(tmp_data.iloc[kk,a*2+1]) != "nan" and str(tmp_data.iloc[kk,a*2+1]) != "None" and str(tmp_data.iloc[kk,a*2+1]) != "0" and tmp_data.iloc[kk,a*2+1] != 0:
                                        effective_date = tmp_date
                                        effective_data = data[data[product_name + "_date"] == effective_date].iloc[kk,a*2+1]
                                        break
                                break

                    tmp_list.append(effective_data)
                    flag = 1

        output[product_name] = tmp_list 
    return output 
