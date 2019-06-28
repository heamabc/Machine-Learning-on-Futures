import pandas as pd
import numpy as np
import datetime

def basis_momentum(sp, fp):
  output = pd.DataFrame()
  output["date"] = fp.iloc[:,0]
  sp = sp.iloc[:,1:]
  fp = fp.iloc[:,1:]

  for j in tqdm(range(fp.shape[1])):
      for r in [5,10,15,20,25,30,35,40,45,50]:
          tmplist = []
          for i in range(r-1):
              tmplist.append(None)

          for i in range(r-1,fp.shape[0]):
              if str(sp.iloc[i-r+1,j]) == "nan" or str(fp.iloc[i-r+1,j]) == "nan":
                  tmplist.append(None)
                  continue           

              tmplist.append(float(sp.iloc[i,j])/float(sp.iloc[i-r+1,j]) - float(fp.iloc[i,j])/float(fp.iloc[i-r+1,j]))

          output[fp_rtn.iloc[:,j].name + "_" + str(r)] = tmplist
