import pandas as pd
import numpy as np

def pct_change(bw, data, periods):
  '''
  calculate backward percentage change
  Parameters
  ----------
  bw: boolean
      If True, calculate backward percentage change. Else, forward percentage change
  data : pd.DataFrame
      The first column must be date, all other columns all data to be calculated.
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
    for j in tqdm(range(data.shape[1])):
        for r in periods:
            tmplist = []
            for i in range(r):
                tmplist.append(None)

            for i in range(r,data.shape[0]):
                if str(data.iloc[i-r,j]) == "nan":
                    tmplist.append(None)
                    continue

                if data.iloc[i-r,j] == 0 and data.iloc[i,j] != 0:
                    tmplist.append(1.0)
                elif data.iloc[i-r,j] == 0 and data.iloc[i,j] == 0:
                    tmplist.append(0.0)
                elif data.iloc[i-r,j] != 0 and data.iloc[i,j] == 0:
                    tmplist.append(-1)
                else:
                    tmplist.append(float(data.iloc[i,j])/float(data.iloc[i-r+1,j]) - 1)

            output[data.iloc[:,j].name + "_" + str(r)] = tmplist
   else:
    for j in tqdm(range(data.shape[1])):
      for r in [5,10,15,20,25,30,35,40,45,50]:
          tmplist = []

          for i in range(data.shape[0]-r):
              if str(data.iloc[i,j]) == "nan":
                  tmplist.append(None)
                  continue
              if data.iloc[i,j] == 0 and data.iloc[i+r,j] != 0:
                  tmplist.append(1.0)
              elif data.iloc[i,j] == 0 and data.iloc[i+r,j] == 0:
                  tmplist.append(0.0)
              elif data.iloc[i,j] != 0 and data.iloc[i+r,j] == 0:
                  tmplist.append(-1)
              else:
                  tmplist.append(float(data.iloc[i+r,j])/float(data.iloc[i,j]) - 1)

          for i in range(r):
              tmplist.append(None)

          output[data.iloc[:,j].name + "_" + str(r)] = tmplist
   return output
   
def med_outlier(data):
  output = pd.DataFrame()
  output["date"] = data["date"]
  data = data.iloc[:,1:]

  for j in range(data.shape[1]):
      median = np.median(data.iloc[:,j])
      median1 = np.median(abs(data.iloc[:,j] - np.mean(data.iloc[:,j])))
      tmplist = []

      for i in range(data.iloc[:,j].shape[0]):
          if data.iloc[i,j] > median + 5*median1:
              tmplist.append(median + 5*median1)
          elif data.iloc[i,j] < median - 5*median1:
              tmplist.append(median - 5*median1)
          else:
              tmplist.append(data.iloc[i,j])

      output[data.iloc[:,j].name] = tmplist
  return output
  
def standardize(data):
