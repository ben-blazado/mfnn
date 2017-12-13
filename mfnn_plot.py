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

def plot_predictions_24hr (predicted_values, target_values, datetimes, str_selected_date):

    #--- convert str_selected_date to datetime format
    selected_date = datetime.strptime(str_selected_date, "%Y-%m-%d").date()
    
    #--- create an array of indexes for the selected date
    dates = [d.date() for d in datetimes]
    idx_selected = np.squeeze (np.argwhere(np.array(dates) == selected_date))
    
    #--- select the values based on idx_selected
    selected_predicted_values = predicted_values[idx_selected]
    selected_target_values    = target_values[idx_selected]
    selected_datetimes        = datetimes[idx_selected]
    
    #--- the values on the x-axis will be the selected date times
    #--- the date times need to be converted to numbers to calculate position of each bar on x-axis
    x_values = date2num(selected_datetimes)

    #--- calculate the width of bar to be used to draw the bar graph on the x-axis
    x_values_width = x_values[-1] - x_values[0]
    num_ticks      = len(x_values)
    tick_width     = 1 / num_ticks * (x_values_width)
    num_bars       = 2  #--- one bar each for predicted and actual values
    bar_width      = (tick_width * 0.8) / num_bars  #--- 2 bars per tick (predicted bar and actual bar)

    #--- draw side-by-side bars for predicted and target values
    _, ax  = plt.subplots(figsize=figsize)
    ax.bar(x_values - bar_width / num_bars, selected_predicted_values, width=bar_width, label='Predicted')
    ax.bar(x_values + bar_width / num_bars, selected_target_values,    width=bar_width, label='Actual')

    #--- format the x axis ticks
    ax.set_xlim(x_values[0] - bar_width, x_values[-1] + bar_width)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    #--- display rest of plot data
    plt.legend()
    plt.title("Predicted vs. Actual Bike Usage (" + selected_date.strftime("%B %d, %Y") + ")")
    plt.xlabel("Hour")
    plt.ylabel("Bike Usage")
    plt.grid(True, axis='y')
    
    return
    

def plot_predictions_bar(predicted_values, target_values, datetimes):

    #---conver datetimest (['date_str', hr]) into datetime format
    datetimes = [datetime.strptime(date_str + ' ' + str(hr), "%Y-%m-%d %H") for [date_str, hr] in datetimes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(True, which='both', axis='y')

    x_values = date2num(datetimes)
    num_ticks = len(x_values)
    tick_width = 1 / num_ticks * (x_values[-1] - x_values[0])
    padding = 5
    bar_width = tick_width / 2 * 0.8 #--- 2 bars per tick (predicted bar and actual bar)
    ax.bar(x_values - bar_width / 2, predicted_values, width=bar_width, label='Predicted')
    ax.bar(x_values + bar_width / 2, target_values, width=bar_width, label='Actual')
    ax.set_xlim(x_values[0] - bar_width, x_values[-1] + bar_width)


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

    
def plot_predictions_line(predicted_values, target_values):

    plt.figure(figsize=figsize)
    plt.plot(target_values, label='Actual')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    return
