import matplotlib.pyplot as plt
#from matplotlib.ticker import MultiplerLocator
from datetime import datetime
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

    #---conver datetimes (['date_str', hr]) into datetime format
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
