import pandas as pd
import numpy as np
import datetime

def basis_momentum(sp, fp, periods):
    '''
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

    for j in range(fp.shape[1]):
        for r in periods:
            tmplist = []
            for i in range(r):
                tmplist.append(None)

            for i in range(r,fp.shape[0]):
                if str(sp.iloc[i-r,j]) == "nan" or str(fp.iloc[i-r,j]) == "nan":
                    tmplist.append(None)
                    continue           

                tmplist.append(float(sp.iloc[i,j])/float(sp.iloc[i-r,j]) - float(fp.iloc[i,j])/float(fp.iloc[i-r,j]))

            output[fp.iloc[:,j].name + "_" + str(r)] = tmplist
    return output


def bias(data, periods):
    '''
    bias: (price - r days moving average) / r days moving average * 100
    '''
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]


    for j in range(data.shape[1]):
        for period in periods:
            ma = data.rolling(period).mean()
            product_name = data.iloc[:,j].name

            output[product_name + "_" + str(period)] = (data.iloc[:,j] - ma.iloc[:,j])/ma.iloc[:,j] * 100
    return output


def ln_price(data):
    output = pd.DataFrame(data.iloc[:,0])
    output = pd.concat([output,np.log(data.iloc[:,1:])], axis=1)
    return output


def return_signal_momentum(data, periods):
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    data = data.pct_change()
    for j in range(data.shape[1]):
        product_name = data.iloc[:,j].name
        for r in periods:
            tmplist = []
            for i in range(data.shape[0]):
                if i < r:
                    tmplist.append(None)
                elif str(data.iloc[i-r,j]) == "nan":
                    tmplist.append(None)
                else:
                    tmpdata = data.iloc[i-r:i,j]
                    count = 0
                    for d in tmpdata:
                        if d >= 0:
                            count += 1
                    tmplist.append(count/r)
            output[product_name + "_" + str(r)] = tmplist
    return output
            

def roll_rtn(sp, fp, ss):
    '''
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
    output['date'] = sp.iloc[:,0]
    
    diff = pd.DataFrame()
    for j in range(1,ss.shape[1]):
        tmplist = []
        for i in range(ss.shape[0]):
            delta = datetime.datetime.strptime(ss.iloc[i,1],"%m/%d/%Y") - datetime.datetime.strptime(ss.iloc[i,0],"%m/%d/%Y")
            tmplist.append(delta.days)
        diff[ss.iloc[:,j].name[:3]] = tmplist
    
    sp = sp.iloc[:,1:]
    fp = fp.iloc[:,1:]
    tmp_output = (np.log(sp) - np.log(fp))*365/diff
    output = pd.concat([output, tmp_output], axis=1)
    return output


def rsi(data, periods):
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
    Return the year-to-year percentage change
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
    Binarize data according to the benchmark.
    Mark 1 if higher than benchmark, mark 0 if lower than benchmark
    '''
    output = pd.DataFrame()
    output['date'] = data.iloc[:,0]
    for j in range(1,data.shape[1]):
        output[data.iloc[:,j].name] = data.iloc[:,j].apply(lambda x: 1 if x>=benchmark else 0 if x < benchmark else None)
        
    return output
