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

  for j in tqdm(range(fp.shape[1])):
      for r in periods:
          tmplist = []
          for i in range(r-1):
              tmplist.append(None)

          for i in range(r-1,fp.shape[0]):
              if str(sp.iloc[i-r+1,j]) == "nan" or str(fp.iloc[i-r+1,j]) == "nan":
                  tmplist.append(None)
                  continue           

              tmplist.append(float(sp.iloc[i,j])/float(sp.iloc[i-r+1,j]) - float(fp.iloc[i,j])/float(fp.iloc[i-r+1,j]))

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
            ma = pd.rolling_mean(data,period)
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

    rtn = data.pct_change()
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
                    tmpdata = rtn.iloc[i-r:i,j]
                    count = 0
                    for d in tmpdata:
                        if d >= 0:
                            count += 1
                    tmplist.append(count/r)
            output[product_name + "_" + str(r)] = tmplist
   return output

def roll_rtn(data, option_code, data):
    '''
    Parameters
    --------------
    data: columns = ["OptionCode", "EndDate", "SettlementDate", "ClosePrice", "CommPrice"]
    '''
    output = pd.DataFrame()
    output["date"] = data[data["OptionCode"] == 305].iloc[:,1]
    
    for option_code in codelist.iloc[:,0]:
    tmplist = []
    tmp_data = data[data["OptionCode"] == option_code]
    if tmp_data.shape[0] < output.shape[0]:
        for abc in range(output.shape[0] - tmp_data.shape[0]):
            tmplist.append(None)
            
    for i in range(tmp_data.shape[0]):
        tdy = datetime.datetime.strptime(tmp_data.iloc[i,1], "%m/%d/%Y")
        settlement = datetime.datetime.strptime(tmp_data.iloc[i,2], "%m/%d/%Y")
        delta = settlement - tdy
        
        tmplist.append((np.log(float(tmp_data.iloc[i,4])) - np.log(float(tmp_data.iloc[i,3])))*365/delta.days)
        
            
        output[option_code] = tmplist
    return output
  
  
def rsi(data, periods):
    output = pd.DataFrame()
    output["date"] = data.iloc[:,0]
    data = data.iloc[:,1:]

    data = data.diff()
    for j in range(data.shape[1]):
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
            output = pd.concat([output,pd.Series(tmplist)], axis=1)
    return output
