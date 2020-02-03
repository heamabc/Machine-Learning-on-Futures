import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

def pct_change(bw, data, periods):
    '''
    Description
    ----------
    calculate backward or foward percentage change
    
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
    data_list = []
    
    def pct_change_function(bw, r, series):
        if series.name == "date":
            return series
        if bw:
            series = series/series.shift(r) - 1
            
            #If the denominator is zero, it will be inf. We convert it into nan
            series.loc[~np.isfinite(series)] = np.nan
            
            return series
        else:
            series = series.shift(-r)/series - 1
            
            #If the denominator is zero, it will be inf. We convert it into nan
            series.loc[~np.isfinite(series)] = np.nan
            
            return series
    
    for r in periods:
        data_list.append(data.apply(lambda x : pct_change_function(bw, r, x)).add_suffix('_' + str(r)))
    
    # Make all the pct change data into 1 df
    data = pd.concat(data_list, axis=1)

    return data

def med_outlier(data):
    '''
    Description
    ----------
    Detect outlier that exceeds median +/- 5*median1, replace with median +/- 5*median1,
    median1 = median of |x - median|
    
    Parameters
    ----------
    data : pd.DataFrame
      The first column must be date, all other columns are data to be calculated.
      
     Returns
    ----------
    pd.DataFrame
      first column is date
    '''

    def med_outlier_function(series):
        if series.name == "date":
            return series
        
        median = np.median(series)
        median1 = np.median( abs( series - np.mean(series) ) )
        
        f = lambda x: median + 5*median1 if x > median + 5*median1 else median - 5*median1 if x < median - 5*median1 else x
        return series.apply(f)
    
    return data.apply(lambda x:med_outlier_function(x))
  
def standardize(data):
    '''
    Description
    ----------
    Standardize all the data to normal distribution N(0,1)
    
    Parameters
    ----------
    data : pd.DataFrame
      The first column must be date, all other columns are data to be calculated.
      
     Returns
    ----------
    pd.DataFrame
      first column is date
    '''

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
    Description
    ----------
    Based on different period(r), generate ranking of all products on each date based on the value. Descending order.
    Eg, ranking of moving average of all products with period 5
    
    Parameters
    ----------
    data : pd.DataFrame
      The first column must be date, all other columns are data to be calculated.
      Same product with different period must be placed adjacently
      
     Returns
    ----------
    pd.DataFrame
      first column is date, return rank df with rank of data with the same period
    '''
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]
    name_list = []

    #count the number of products to decide number of groups (how many different periods are there)
    name_list = [name[:3] for name in data.columns]
    length = len(np.unique(name_list))
    num_groups = int(data.shape[1]/length)

    for i in range(num_groups):
        # Make sub group according to their period
        sub_data = data.iloc[:,i * length : i * length -1]

        # Rank among sub group
        sub_data = sub_data.rank(axis=1,method="first", ascending=False)
        output = pd.concat([output,sub_data], axis=1)
        
    return output


def match_date(data, date):
    '''
    Description
    ----------
    Date of the raw data is different. This function is the merge by their date.
    Handle many of the anormal cases with the data.
    
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

    for a in range(int(data.shape[1]/2)):
        product_name = str(data.iloc[:,a*2].name[0:3])
        tmp_list = []
        flag = 0
        data_count = data.iloc[:,a*2].shape[0] - data.iloc[:,a*2].isnull().sum()
        first_date = datetime.datetime.strptime(data.iloc[0,a*2], "%m/%d/%Y")
        
        for i in range(date.shape[0]):       

            today = datetime.datetime.strptime(date.iloc[i,0], "%m/%d/%Y")
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
                            j -= 1
                            break
                        tmp_date = datetime.datetime.strptime(data.iloc[index + j,a*2], "%m/%d/%Y")
                        delta = tmp_date - today
                        if delta.days > 0:
                            break

                    if str(data.iloc[index + j,a*2+1]) != "nan" and str(data.iloc[index + j,a*2+1]) != "None" and str(data.iloc[index + j,a*2+1]) != "0" and data.iloc[index+j,a*2+1] != 0:
                        delta_list.append(delta.days)

                    if len(delta_list) > 0:
                        tmp_date = today + datetime.timedelta(np.max(delta_list))
                        tmp_date = '{d.month}/{d.day}/{d.year}'.format(d=tmp_date)
                        tmp_data = data[data[product_name + "_date"] == tmp_date]

                        #if today has multiple data, use only the first useful data. If no useful data, effective date
                        #and effective data remain unchanged
                        for kk in range(tmp_data.shape[0]):
                            if str(tmp_data.iloc[kk,a*2+1]) != "nan" and str(tmp_data.iloc[kk,a*2+1]) != "None" and str(tmp_data.iloc[kk,a*2+1]) != "0" and tmp_data.iloc[kk,a*2+1] != 0:
                                effective_date = tmp_date
                                effective_data = data[data[product_name + "_date"] == effective_date].iloc[kk,a*2+1]
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

                        tmp_date = today + datetime.timedelta(np.max(delta_list))
                        tmp_date = '{d.month}/{d.day}/{d.year}'.format(d=tmp_date)
                        tmp_data = data[data[product_name + "_date"] == tmp_date]

                        #if today has multiple data, use only the first useful data. If no useful data, effective date
                        #and effective data remain unchanged
                        for kk in range(tmp_data.shape[0]):
                            if str(tmp_data.iloc[kk,a*2+1]) != "nan" and str(tmp_data.iloc[kk,a*2+1]) != "None" and str(tmp_data.iloc[kk,a*2+1]) != "0" and tmp_data.iloc[kk,a*2+1] != 0:
                                effective_date = tmp_date
                                effective_data = data[data[product_name + "_date"] == effective_date].iloc[kk,a*2+1]
                                break
                                
                    tmp_list.append(effective_data)
                    flag = 1

        output[product_name] = tmp_list 
    return output 
