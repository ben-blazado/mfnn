import pandas as pd
import numpy as np
from mfnn_utils import retrieve, unzip


class BikeShareData:
    
    
    def __init__(self,
                 url     = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip',
                 data_dir='data',
                 filename_hour_data = 'hour.csv'):
        
        self.url                 = url
        self.data_dir            = data_dir
        self.filename_hour_data  = filename_hour_data
        
        self.df_bike_share_data  = None   
        #--- stores all bike sharing data 
        #--- will be broken up into training, validation, and testing sets
        
        self.df_bike_share_dates = None
        #--- stores the dates of each bike share data
        #--- will be referenced displaying the predictions of test data
        
        self.unused_cols      = ['instant', 'dteday', 'atemp', 'workingday', 'yr']
        #--- unused_cols is a list of names of columns that will not be used in training
        
        self.categorical_cols = ['season', 'weathersit', 'mnth', 'hr', 'weekday', 'holiday']
        self.continuous_cols  = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        self.z_factors        = {}
        self.target_cols      = ['cnt']
        self.date_col         = ['dteday']

        self.ax_column        = 1 #--- constant for axis number of columns in dataframe
        
        self.days_for_test_data       = 21    
        self.days_for_validation_data = 60

        return
    
    def retrieve(self):
        
        file_name = retrieve(url=self.url, dest_dir=self.data_dir)
        unzip(file_name, self.data_dir)
        
        #--- create the pandas data frame for the bike share data from the unzip .csv file
        self.df_bike_share_data = pd.read_csv(self.data_dir + '/' + self.filename_hour_data)
        return
    
    def preprocess(self):
        
        self.create_ohe_cols()
        self.rescale_cont_cols()
        self.drop_unused_cols()
        
        return
    
    
    def create_data_sets(self):
        
        training_set, validation_set = self.create_training_set()
        testing_set                  = self.create_testing_set()

        return training_set, validation_set, testing_set
    
    
    def create_ohe_cols(self):

        #--- one hot encode (ohe) categorical values
        #--- TODO: explain design decision in documents (i.e. which categorical fields, why ohe?)
        for col in self.categorical_cols:
            dummies = pd.get_dummies(self.df_bike_share_data[col], prefix=col, drop_first=False)
            self.df_bike_share_data = pd.concat([self.df_bike_share_data, dummies], axis=self.ax_column)

        #--- remove the categorical columns since they are now one-hot-encoded
        self.df_bike_share_data = self.df_bike_share_data.drop(self.categorical_cols, axis=self.ax_column)

        return 

    def drop_unused_cols(self):
        #--- drop columns that network will not use during training
        #--- TODO: explain design decision (i.e. which fields and why)
        #--- TODO: change axis to AX_COLUMNS

        self.df_bike_share_dates = self.df_bike_share_data[self.date_col]
        #--- save the dates column for use in testing before dropping
        
        self.df_bike_share_data = self.df_bike_share_data.drop(self.unused_cols, axis=self.ax_column)

        return
    
    
    def rescale_cont_cols(self):

        #--- do not call more than once
        #--- rescale continuous values to a z-score: z[i] = (x[i] - x_mean) / x_std_dev
        #--- use the z_factors dictionary to store mean and std_dev for each col
        
        for col in self.continuous_cols:
            mean    = self.df_bike_share_data[col].mean()
            std_dev = self.df_bike_share_data[col].std()
            
            self.df_bike_share_data[col] = (self.df_bike_share_data[col] - mean) / std_dev
            self.z_factors[col]          = (mean, std_dev)

        return
    
    
    def create_training_set(self):
        
        #--- keep ALL UP TO the last n days [:-n*24] of data for training the network
        df_training_data = self.df_bike_share_data[:-self.days_for_test_data * 24]
        
        #--- split training data into features and targets
        df_features   = df_training_data.drop(self.target_cols, axis=self.ax_column)
        df_targets    = df_training_data[self.target_cols]
        
        #--- split features and targets into sets used for training and validation of the training
        idx_validation_data_split = self.days_for_validation_data * 24    
        
        #--- keep ALL UP TO the last n days [:-n] of features and targets for training data 
        df_training_features   = df_features[:-idx_validation_data_split]
        df_training_targets    = df_targets[:-idx_validation_data_split]
        
        #--- keep THE LAST n days [-n:] for validation (during training)
        df_validation_features = df_features[-idx_validation_data_split:]
        df_validation_targets  = df_targets[-idx_validation_data_split:]

        #--- create the training and validation sets as tuples of their respective features and targets
        training_set           = (df_training_features, df_training_targets)
        validation_set         = (df_validation_features, df_validation_targets)
        
        return training_set, validation_set
    
    
    def create_testing_set(self):
        
        #--- keep THE LAST n days [-n*24:] of data for testing the neural network
        df_testing_data  = self.df_bike_share_data[-self.days_for_test_data * 24:]
        
        #--- reference the dates of the test data
        df_test_dates = self.df_bike_share_dates[-self.days_for_test_data * 24:]
        
        #--- split testing data into features and targets
        df_test_features = df_testing_data.drop(self.target_cols, axis=self.ax_column)
        df_test_targets  = df_testing_data[self.target_cols]
        
        #--- create the test set as tuples of the test features and test targets
        testing_set = (df_test_features, df_test_targets, df_test_dates)
        
        return testing_set
    

