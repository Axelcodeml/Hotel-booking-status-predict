#!/usr/bin/env python
# coding: utf-8


# Script of data preparation
import pandas as pd
import numpy as np
import os


# functions to treat outliers by flooring and capping
def treat_outliers(df, col):
    Q1 = df[col].quantile(0.25)  # 25th quantile
    Q3 = df[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1
    Low_Whisker = Q1 - 1.5 * IQR
    Upp_Whisker = Q3 + 1.5 * IQR

    # all the values smaller than Lower_Whisker will be assigned the value of Lower_Whisker
    # all the values greater than Upper_Whisker will be assigned the value of Upper_Whisker
    df[col] = np.clip(df[col], Low_Whisker, Upp_Whisker)


def treat_outliers_all(df, col_list):
    for c in col_list:
        df = treat_outliers(df, c)

    return df


# Read csv files:
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, 'Succesfully loaded')
    return df


#We do the data transformation
def data_preparation(df):
    # Changing type of data
    df['required_car_parking_space'] = df['required_car_parking_space'].astype("object") 
    df['repeated_guest'] = df['repeated_guest'].astype("object")
    df['arrival_year'] = df['arrival_year'].astype("object")
    df['arrival_month'] = df['arrival_month'].astype("object")
    
    #Changing by values of yes or no
    index_aux=df[df['required_car_parking_space']==1].index.tolist()
    df.loc[index_aux,'required_car_parking_space']='Yes'
    index_aux=df[df['required_car_parking_space']==0].index.tolist()
    df.loc[index_aux,'required_car_parking_space']='No'

    index_aux=df[df['repeated_guest']==1].index.tolist()
    df.loc[index_aux,'repeated_guest']='Yes'
    index_aux=df[df['repeated_guest']==0].index.tolist()
    df.loc[index_aux,'repeated_guest']='No'
    
    df["booking_status"] = df["booking_status"].apply(lambda x: 1 if x == "Canceled" else 0)
    
    #Changing string values in months
    years=['2017','2018']
    months=['January','February','March','April','May','June','July','August','September','October','November','December']

    y_num=2017
    for y_text in years:
        index_aux=df[df['arrival_year']==y_num].index.tolist()
        df.loc[index_aux,'arrival_year']=y_text
        y_num=y_num+1
    
    m_num=1
    for m_text in months:
        index_aux=df[df['arrival_month']==m_num].index.tolist()
        df.loc[index_aux,'arrival_month']=m_text
        m_num=m_num+1    

    #Treatment of very skewed data with logarithms
    df["no_of_week_nights_log"] = np.log(df["no_of_week_nights"]+1)        
    df["lead_time_log"] = np.log(df["lead_time"]+1)        

    #Treatment of outliers
    treat_outliers(df,'lead_time')
    treat_outliers(df,'avg_price_per_room')    
    
    # Transformation to category
    df['Booking_ID'] = df['Booking_ID'].astype('category')
    df['type_of_meal_plan'] = df['type_of_meal_plan'].astype('category')
    df['required_car_parking_space'] = df['required_car_parking_space'].astype('category')
    df['room_type_reserved'] = df['room_type_reserved'].astype('category')
    df['arrival_year'] = df['arrival_year'].astype('category')
    df['arrival_month'] = df['arrival_month'].astype('category')
    df['market_segment_type'] = df['market_segment_type'].astype('category')
    df['repeated_guest'] = df['repeated_guest'].astype('category')
    df['booking_status'] = df['booking_status'].astype('int64')
    
    print('Completed data transformation')
    
    return df


def data_exporting(df, filename):
    dfp = df
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'Correctly exported to processed folder')


def main():
    # Training data
    df1 = read_file_csv('bookingcc.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,'booking_train.csv')
    
    # Validation data
    df2 = read_file_csv('bookingcc_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,'booking_val.csv')
    
    # Scoring data
    df3 = read_file_csv('bookingcc_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3,'booking_score.csv')


if __name__ == "__main__":
    main()
