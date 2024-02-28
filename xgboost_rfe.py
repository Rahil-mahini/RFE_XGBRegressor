# -*- coding: utf-8 -*-

import sys
print (sys.path)
import os
import pandas as pd
from dask.distributed import Client , LocalCluster
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE 


        

# Load data from X_file_path csv file and return dataframe
def load_X_data(X_file_path):
    
    try:
        # Load data from the CSV file which include the header as pandas dataframe
        df = pd.read_csv(X_file_path, sep=','  ) 
        print ( "X dataframe shape ", df.shape)
        
        
        # df = df.iloc [:, :30]  
        # print ( "X dataframe shape ", df.shape)
        # print ( "X dataframe " , df)
        
        
        return df

    except Exception as e:
        print("Error occurred while loading descriptors CSV data:", e)
        
        
        
# Load data from y_file_path csv file and return dask cudf       
def load_y_data(y_file_path):
    
    try:
       
        # Load data from the CSV file which include the header using pandas dataframe  
        df = pd.read_csv(y_file_path, sep= ','    )
        
        # exclude the first column if includes the samples' names
        df = df.iloc[:, 1:]
      
        print ( "y dataframe shape", df.shape)
        print ( "y dataframe", df)
        

        
        return  df

    except Exception as e:
        print("Error occurred while loading concentrations CSV data:", e)
        
        
def feature_importacne(X_file_path, y_file_path, num_feature_to_select):
    
    X = load_X_data(X_file_path)
    y = load_y_data(y_file_path)

    
    X = X.astype('float32')  # Features MUST be float32
    y = y.astype('float32')  # Labels MUST be float32

     # Create an XGBoost model
    model = XGBRegressor()
    
    rfe = RFE ( estimator=model, n_features_to_select=num_feature_to_select)  
 
    
    # Fit SelectKBest on the data
    rfe.fit(X, y)
        
     
    # Get the names of the selected features
    selected_feature_names = X.columns[rfe.support_]
    print("selected_feature_names type :" , type(selected_feature_names))
    print("selected_feature_names contain :" , selected_feature_names)
    
     # Retrieve the selected features from the input with their column names
    selected_features = X[selected_feature_names]
    print("selected_features type :" , type(selected_features))
    print("Selected Feature Names:", selected_feature_names)
    print("Selected Features Dataframe:", selected_features)    
    print ("selected columns " , selected_features.columns)
    

    
    return selected_features



# Function gets the pandas dataframe write it to csv file and return file path dictionaty
def write_to_csv(X_selected, output_path):
    

    
    # Convert the list to pandas dataframe 
    table_df = pd.DataFrame(X_selected)
    
    
    # Create a separate directory for the output file
    try:
        
      # Create the output directory if it doesn't exist                                                        
       os.makedirs(output_path, exist_ok = True)     
       file_name = 'combinatorial_selected_xgboost_kbest.csv'
       file_path = os.path.join(output_path, file_name)
                
       table_df.to_csv(file_path, sep = ',', header =True, index = True ) 

       file_path_dict = {'combinatorial_xgboost_kbest': file_path}
       print("CSV file written successfully.")
       print ("CSV file size is  " , os.path.getsize(file_path))
       print ("CSV file column number is  " , table_df.shape[1])
       print ("file_path_dictionary is  " , file_path_dict)
       return file_path 
   
    except Exception as e:
       print("Error occurred while writing matrices to CSV:", e)
       
       
if __name__ == '__main__': 
         
                 
    # Create Lucalluster with specification for each dask worker to create dask scheduler     
    cluster = LocalCluster (n_workers= 1,  threads_per_worker= 28, memory_limit='460GB',  timeout= 3000)
    
    #Create the Client using the cluster
    client = Client(cluster) 

       
    X_file_path = r'/features.csv' 
    y_file_path = r'/endpoint.csv'       
    output_path = r'output'
    
    num_feature_to_select = 1000
    X_selected = feature_importacne(X_file_path, y_file_path, num_feature_to_select)
    file_path = write_to_csv(X_selected, output_path)
    print (file_path)
 
    scheduler_address = client.scheduler.address
    print("Scheduler Address:", scheduler_address)
    print(cluster.workers)
    print(cluster.scheduler)
    print(cluster.dashboard_link)
    print(cluster.status)
   
   
    client.close()        
