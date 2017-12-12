import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from  matplotlib.dates import date2num
#from matplotlib.ticker import MultiplerLocator
from datetime import datetime
from datetime import timedelta  
import numpy as np

figsize = (10,6)

def plot_losses (training_losses, validation_losses):
    #--- display the losses on a graph
    plt.figure(figsize=figsize)
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)        
    plt.title("Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return

    
def plot_predictions_bar(predicted_values, target_values, datetimes):

    #---conver datetimest (['date_str', hr]) into datetime format
    datetimes = [datetime.strptime(date_str + ' ' + str(hr), "%Y-%m-%d %H") for [date_str, hr] in datetimes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(True, which='both', axis='y')

    x_values = date2num(datetimes)
    bar_width = (1/len(predicted_values)) / 2
    ax.bar(x_values - bar_width /2.5 , predicted_values, width=bar_width/1.25, label='Predicted')
    ax.bar(x_values + bar_width /2.5, target_values, width=bar_width/1.25, label='Actual')
    ax.set_xlim(x_values[0] - bar_width /1.25, x_values[-1] + bar_width /1.25)


    ax.xaxis_date()
    ax.xaxis.set_tick_params(which='major', pad=15)    
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
    fig.autofmt_xdate(rotation=90, ha='center')    
    
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date/Hour")
    plt.ylabel("Bike Usage")
    
    return

    
def plot_predictions_bar_old(predicted_values, target_values, datetimes):

    #---conver datetimest (['date_str', hr]) into datetime format
    datetimes = [datetime.strptime(date_str + ' ' + str(hr), "%Y-%m-%d %H") for [date_str, hr] in datetimes]
    x_values = np.arange(len(predicted_values))
    bar_width = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    
    # plt.figure(figsize=figsize)
    plt.bar(x_values, predicted_values, width=bar_width, label='Predicted')
    plt.bar(x_values + bar_width, target_values, width=bar_width, label='Actual')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    plt.xticks(x_values + bar_width / 2, datetimes, rotation='vertical')
    
    return


def plot_predictions_line(predicted_values, target_values):

    plt.figure(figsize=figsize)
    plt.plot(target_values, label='Actual')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    return
