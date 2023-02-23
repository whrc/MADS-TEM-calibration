import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import statistics as stat
import math
#-------------------------------FUNCTIONS TO LOAD CSV FILES----------------------------------------------------

#read model data from csv file
#returns dict: {'obs_id':[model values]}
def read_csv_model(path,filename):
  mod={}
  with open(path+filename, 'r') as file:
      reader = csv.reader(file)
      r=1
      for row in reader:
          vals=[]
          if r==1:
            r=r+1
          else:
            for nn in row[1:]:
              vals.append(nn)
            mod[row[0]]=vals
  return mod

#read param data from param csv file
#returns dict: {'param':[optimal param values]}
def read_csv_params(path,filename):
  mod={}
  with open(path+filename, 'r') as file:
      reader = csv.reader(file)
      r=1
      for row in reader:
          vals=[]
          if r==1:
            r=r+1
          else:
            for nn in row[5:]:
              vals.append(nn)
              vals=[float(x) for x in vals]
            mod[row[0]]=vals
  return mod

#read error data from param csv file
#returns list of errors as strings
def read_csv_errors(path,filename):
  with open(path+filename, 'r') as file:
      reader = csv.reader(file)
      r=1
      for row in reader:
        if r==1:
          r=r+1
          vals=row[5:]
      #remove 'OF:' left over from iteration files
      for nn in range(0,len(vals)):
        vals[nn]=vals[nn].replace("OF:", "")
        vals[nn]=vals[nn].replace("_1", "")
        vals[nn]=vals[nn].replace("_2", "")
        vals[nn]=vals[nn].replace("_3", "") 
  return vals

#-------------------------------FUNCTIONS TO LOAD ITERATION FILES----------------------------------------------

def get_optimal_sets_of_params(filename):
  #we assume there are three lines per calibration - 0=OF, 1=lambda, 2=params
  with open(filename) as f:
      lines = f.readlines()
  #for multiple optimal sets, need to loop through them all, index starts at 0
  filelength = len(lines)
  num_sets = math.floor(filelength/3) #truncate in case there's an empty extra line at end of file
  for nn in range(1,num_sets+1):
      del lines[(nn-1)] #delete OF line
      del lines[(nn-1)] #delete lambda line
  #remove formatting from iteration files
  for nn in range(0,num_sets):
      lines[nn]=lines[nn].replace("OrderedCollections.OrderedDict", "") 
      lines[nn]=lines[nn].replace(" ", "")
      lines[nn]=lines[nn].replace("\"", "")
      lines[nn]=lines[nn].replace("\n", "")
      lines[nn]=lines[nn].replace("(", "")
      lines[nn]=lines[nn].replace(")", "")
      lines[nn]= dict(subString.split("=>") for subString in lines[nn].split(","))
  # MERGE OPTIMAL VALUES INTO ONE KEY/VALUE SET IN DICTIONARY:
  #at this point, lines is a set of key:value pairs of optimal sets for each calibration run
  #we combine all runs into one set of keys (params) with multiple optimal value to plot easier:
  params = {}
  for sub in lines:
    for key, val in sub.items(): 
      params.setdefault(key, []).append(round(float(val),2))
  return params

def merge_parameter(p1,p2):
  merge_param = {**p1, **p2}
  for same_key in set(p1) & set(p2):
    merge_param[same_key] = p1[same_key]+p2[same_key]
  return merge_param

#read in error from file (file name passed in as arg)
def get_error(filename):
  #again, assumes there are three lines - 0=OF, 1=lambda, 2=params
  with open(filename) as f:
    errors = f.readlines()
  #for multiple optimal sets, need to loop through them all
  filelength = len(errors)
  num_sets = math.floor(filelength/3) #truncate in case there's an empty extra line at end of file
  for nn in range(1,num_sets+1):
    del errors[(nn)] #delete lambda line
    del errors[(nn)] #delete params line
  for nn in range(0,num_sets):
    errors[nn]=errors[nn].replace("OF:", "") 
    errors[nn]=errors[nn].replace(" ", "")
    errors[nn]=errors[nn].replace("\"", "")
    errors[nn]=errors[nn].replace("\n", "")
  return errors

#Load error from iteration results file and identify separate runs 
def load_sort_itr_err(path,filename):
  #load iteration errors:
  e_itr=get_error(path+filename)
  float_err_itr=[float(x) for x in e_itr]
  rounded_err_itr=list(np.round(float_err_itr,7))
  #find jumps in error to identify different calibration runs:
  diff=[t - s for s, t in zip(rounded_err_itr, rounded_err_itr[1:])]
  Q1 = np.percentile(diff, 25, interpolation = 'midpoint')
  Q3 = np.percentile(diff, 75, interpolation = 'midpoint')
  IQR = Q3 - Q1
  upper = np.where(diff >= (Q3+1.5*IQR)) # index of error jumps greater than upper bound of IQR
  #Split iteration file error data into separate calibration runs:
  idx=list(upper[0])
  idx=[x+1 for x in idx] #the upper var is based on differences in error, we need this index +1
  idx.append(len(rounded_err_itr))
  err_by_run=[rounded_err_itr[x:y] for x,y in zip([0]+idx[:-1],idx)]
  return rounded_err_itr,idx,err_by_run

#------------------------------------PLOTTING FUNCTIONS-------------------------------------------------------------

def plot_histograms(params,nbins=10,x=16,y=8,r=2,c=4):
  #plot the optimal values
  plt.figure(figsize=(x,y))
  s=1
  for item in params:
    plt.subplot(r,c,s)
    plt.hist(params[item],bins=nbins);
    plt.title(item)
    plt.xlabel('optimal values')
    plt.ylabel('counts')
    s+=1
  return

#Use Kmeans to cluster errors, needs input of list errors converted to floats
def get_err_clusters(float_err,n_clusters=4):
  arr=np.array(float_err)
  kmeans = KMeans(n_clusters)
  kmeans.fit(arr.reshape(-1,1))
  y_kmeans = kmeans.predict(arr.reshape(-1,1))
  centers = kmeans.cluster_centers_
  centers = sorted(centers) #do we need this line?
  return y_kmeans, centers

#Organize parameters values by kmeans clusters
def cluster_param_data(params,y_kmeans):
  zeroes=[]
  ones=[]
  twos=[]
  threes=[]
  for v in range(len(params)):
    if y_kmeans[v]==0:
      zeroes.append(params[v])
    elif y_kmeans[v]==1:
      ones.append(params[v])
    elif y_kmeans[v]==2:
      twos.append(params[v])
    elif y_kmeans[v]==3:
      threes.append(params[v])
  return zeroes, twos, ones, threes

#Stacked Plot: uses kmeans centers to plot histogram by error groups
def plot_stacked_histograms(mparams,centers,y_kmeans,nbins=10,x=16,y=8,r=2,c=4,std=0):
  #plot the optimal values, colors stacked by error clusters
  plt.style.use('bmh')
  plt.figure(figsize=(x,y))
  labels=[float(x) for x in centers]
  rounded_labels=list(np.round(labels,4))
  s=1
  for item in mparams:
    if stat.stdev(mparams[item])>std:
      plt.subplot(r,c,s)
      zeroes,ones,twos,threes=cluster_param_data(mparams[item],y_kmeans)
      plt.hist([zeroes,ones,twos,threes], 10, density=False, histtype='bar', stacked=True)
      plt.title(item)
      plt.xlabel('optimal values')
      plt.ylabel('counts')
      plt.tight_layout(pad=3.0)
      s+=1
    else:
      print('Parameter '+item+' has standard deviation('+str(stat.stdev(mparams[item]))+') less than threshold of '+str(std)+'\n')
  centers=np.round(centers,2)
  # plt.legend(['Type I','Type II','Type III','Type IV'], title = "Error", bbox_to_anchor=(1.0, 1.0), loc='upper right')
  plt.legend([str(centers[0]),str(centers[1]),str(centers[2]),str(centers[3])], title = "Error", bbox_to_anchor=(1.0, 1.0), loc='upper right')
  plt.suptitle('Optimal Parameters Classified by Errors')
  return

#Plot error results from iteration file in separate subplot for each run, include polynomial fit to data
def plot_err_by_run(path,filename,err_by_run, idx, x=16, y=8, r=3, c=4, deg=2):
  #Using split iteration file data, plot error by iteration with polynomial fit(default deg=2):
  #load iteration errors:
  plt.style.use('bmh')
  plt.figure(figsize=(x,y))
  s=1
  for i in (range(len(idx))):
    plt.subplot(r,c,s)
    plt.plot(np.log10(err_by_run[i]), label='Iteration Error');
    plt.xlabel('Iteration number')
    plt.ylabel('Error (log scale)')
    plt.title('Calibration run:' + str(i+1))
    # fit polynomial to data on log scale:
    num_itr=len(err_by_run[i])
    x_ax=list(range(0,num_itr))
    y_fit=np.polyfit(x_ax, np.log10(err_by_run[i]), deg)
    y=np.poly1d(y_fit)
    x_fit=np.linspace(0,num_itr-1,20)
    plt.plot(x_fit, y(x_fit),'-', label='Fitted curve')
    plt.tight_layout()
    # plt.legend()
    s+=1
  plt.suptitle('Error Evolution per Iteration for Each Calibration Run')
  return

def plot_err(err):
  float_err=[float(x) for x in err]
  y_kmeans,centers=get_err_clusters(float_err)
  plt.figure()
  plt.scatter([i for i in range(len(float_err))], float_err, c=y_kmeans)
  # plt.plot(rounded_err,'o')
  plt.xlabel('Calibration Run')
  plt.ylabel('Error')
  plt.title('Final error for each run')
  return

