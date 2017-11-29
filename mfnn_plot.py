import matplotlib.pyplot as plt
import numpy as np

def plot_losses (training_losses, validation_losses):
    #--- display the losses on a graph
    plt.figure(figsize=(10,6))
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)        
    plt.title("Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return


def plot_predictions_bar(predicted_values, target_values):

    x_values = np.arange(len(predicted_values))
    bar_width = 0.4
    
    plt.figure(figsize=(10,6))
    plt.bar(x_values, predicted_values, width=bar_width, label='Predicted')
    plt.bar(x_values + bar_width, target_values, width=bar_width, label='Actual')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    plt.xticks(x_values + bar_width / 2, x_values)
    
    return


def plot_predictions_line(predicted_values, target_values):

    plt.figure(figsize=(10,6))
    plt.plot(target_values, label='Actual')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    return
