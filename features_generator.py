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

