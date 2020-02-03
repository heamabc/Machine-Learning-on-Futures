import pandas as pd
import numpy as np
import datetime

def basis_momentum(sp, fp, periods):
    '''
    Description
    ----------
    basis momentum: the difference of r days culmulative return of nearby futures and farther futures.
    
    Parameters
    ----------
    sp: pd.DataFrame
      first column is date, all other columns are nearby futures price.
    fp : pd.DataFrame
      first column is date, all other columns are farther futures price.
    periods: list
      list of periods of basis momentum to be calculated
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are basis momentum
    '''
    output = pd.DataFrame()
    output["date"] = fp.iloc[:,0]
    sp = sp.iloc[:,1:]
    fp = fp.iloc[:,1:]    
    
    for r in periods:   
        for j in range(fp.shape[1]):
            tmp = np.where(sp.iloc[:,j].notnull() & fp.iloc[:,j].notnull(), 
                           sp.iloc[:,j].astype(float)/sp.iloc[:,j].shift(r).astype(float) - fp.iloc[:,j].astype(float)/fp.iloc[:,j].shift(r).astype(float),
                           None)
            
            tmp[:r] = r * [None]

            output[fp.iloc[:,j].name + "_" + str(r)] = tmp
    return output


def bias(data, periods):
    '''
    Description
    ----------
    bias: (price - r days moving average) / r days moving average * 100
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are futures spot price.
    periods: list
      list of periods of bias to be calculated
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are basis momentum
    '''
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    def bias_generator(series):
         
        df = pd.DataFrame(len(periods) * [series]).T
        
        for period in periods:
            df[series.name + '_' + str(period)] = series.rolling(period).mean()
        
        price = df.iloc[:,:len(periods)]
        ma = df.iloc[:,len(periods):]
        
        return (price.values - ma) / ma * 100
    
    df_list = []
    for col in data:
        df_list.append(bias_generator(data[col]))

    return pd.concat([output,pd.concat(df_list, axis=1)], axis=1)


def ln_price(data):
    '''
    Description
    ----------
    natural log of price
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are futures spot price.

    Returns
    ----------
    pd.DataFrame
      first column is date, all other are ln price
    '''
    
    output = pd.DataFrame(data.iloc[:,0])
    output = pd.concat([output,np.log(data.iloc[:,1:])], axis=1)
    return output


def return_signal_momentum(data, periods):
    '''
    Description
    ----------
    for r days, if there are more positive one-day return than negative one-day return, 1, else 0
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are futures spot price.
    periods: list
      list of periods of return signal momentum to be calculated
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are return signal momentum
    '''
    
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]
    df_list = []

    data = data.pct_change()
    
    def return_signal_momentum_function(r, series):
        tmp = pd.Series(0)
        
        for i in range(r):
            tmp += np.where(series.shift(r) >=0, 1, -1)
            if i == r-1:
                na_count = series.shift(r).isnan().sum()
                
        tmp.loc[tmp.iloc[:na_count].index] = None
        tmp.loc[tmp.iloc[:r].index] = None
        
        tmp = np.where(tmp >= 0, 1,
                 np.where(tmp.isnan()), None,
                 0)
        
        return tmp
        
    for r in periods:
        df_list.append(data.apply(lambda x:return_signal_momentum_function(r, x)).add_suffix('_' + str(r)))
        
    data = pd.concat(df_list, axis=1)
        
    return pd.concat([output, data], axis=1)
            

def roll_rtn(sp, fp, ss):
    '''
    Description
    ------------
    (log(sp) - log(fp))*365/diff
    diff = settlement date - today
    
    Parameters
    ------------
    sp: pd.DataFrame
      1 col = date, all other col are corresponding spot price
    fp : pd.DataFrame
      1 col = date, all other col are corresponding futures price
    ss: pd.DataFrame
      1 col = date, all other col are corresponding settlement date
      
    Return
    -------------
    pd.DataFrame
      1 col = date, all other col are corresponding roll return
    '''
    output = pd.DataFrame()
    ss = ss.iloc[:,1:]
    output['date'] = sp.iloc[:,0]

    def diff_function(series):
        series = pd.to_datetime(series)
        date = pd.to_datetime(output['date'])
        
        return (series - date).days
            
    diff = ss.apply(lambda x:diff_function(x))
    
    sp = sp.iloc[:,1:]
    fp = fp.iloc[:,1:]
    tmp_output = (np.log(sp) - np.log(fp))*365/diff
    output = pd.concat([output, tmp_output], axis=1)
    return output


def rsi(data, periods):
    '''
    Description
    ----------
    for r days, if there are more positive one-day return than negative one-day return, 1, else 0
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are futures spot price.
    periods: list
      list of periods of rsi to be calculated
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are rsi
    '''
    
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    data = data.diff()
    for j in range(data.shape[1]):
        product_name = str(data.iloc[:,j].name)
        for period in periods:
            tmplist = []
            a_gain = []
            a_loss = []
            flag = 0
            for i in range(data.shape[0]):
                if i < period:
                    tmplist.append(None)
                elif str(data.iloc[i,j]) == "nan":
                    tmplist.append(None)
                elif str(data.iloc[i-period,j]) == "nan":
                    tmplist.append(None)
                else:
                    up = []
                    down = []
                    for k in range(period):
                        if data.iloc[i-k,j] >= 0:
                            up.append(data.iloc[i-k,j])
                        else:
                            down.append(data.iloc[i-k,j])

                    tmplist.append( (np.sum(up)/period) /((np.sum(up)/period) + abs(np.sum(down)/period) )*100)
            output = pd.concat([output,pd.Series(tmplist, name = product_name + "_" + str(period))], axis=1)
    return output


def seasonality(data): 
    '''
    Description
    ----------
    Return the year-to-year percentage change
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are futures spot price.
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are year-to-year pct change
    '''
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]

    for j in range(1,data.shape[1]):
        first_date = datetime.datetime.strptime(data.iloc[data.iloc[:,j].first_valid_index(),0],"%m/%d/%Y")
        last_date = datetime.datetime.strptime(data.iloc[-1,0],"%m/%d/%Y")
        tmplist = []
        for i in range(data.shape[0]):
            today = datetime.datetime.strptime(data.iloc[i,0], "%m/%d/%Y")

            if today < first_date:
                tmplist.append(np.nan)
                continue
            else:
                delta = today - first_date
                if delta.days < 365:
                    tmplist.append(np.nan)
                    continue

                delta = last_date - today
                if delta.days < 0:
                    tmplist.append(np.nan)
                    continue

                delta_list = []

                if today.month == 2 and today.day == 29:
                    last_year = today.replace(year = today.year - 1,day = 28)
                else:
                    last_year = today.replace(year = today.year - 1)
                if i - 252 -20< 0:
                    minn = 252 - i
                else:
                    minn = -20
                for abc in range(minn,21):
                    tmp_date = datetime.datetime.strptime(data.iloc[i-252+abc,0], "%m/%d/%Y")
                    delta = tmp_date - last_year
                    delta_list.append(abs(delta.days))

                index = np.argmin(delta_list)
                if data.iloc[i,j] != 0 and data.iloc[i-252+ minn + index[0],j] == 0:
                    if data.iloc[i,j] > 0:
                        tmplist.append(1)
                    else:
                        tmplist.append(-1)
                elif data.iloc[i,j] == 0 and data.iloc[i-252+minn+index[0],j] == 0:
                    tmplist.append(0)
                else:
                    tmplist.append(data.iloc[i,j]/data.iloc[i-252+ minn + index[0],j] - 1)  
        output[data.iloc[:,j].name] = tmplist
    return output


def signalizer(data, benchmark):
    '''
    Description
    ----------
    Binarize data according to the benchmark.
    Mark 1 if higher than benchmark, mark 0 if lower than benchmark
    
    Parameters
    ----------
    data: pd.DataFrame
      first column is date, all other columns are values.
    benchmark: int/float
      benchmark value
      
    Returns
    ----------
    pd.DataFrame
      first column is date, all other are Binarized data
    '''
    output = pd.DataFrame()
    output['date'] = data.iloc[:,0]
    
    def signalizer_function(benchmark, series):
        return series.apply(lambda x: 1 if x>=benchmark else 0 if x < benchmark else None)
        
    data = data.iloc[:,1:].apply(lambda x:signalizer_function(benchmark, series))
    return pd.concat([output,data], axis=1)
