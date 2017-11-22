import matplotlib.pyplot as plt

def plot_losses (training_losses, validation_losses):
    #--- display the losses on a graph
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)        
    plt.title("Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.figure(figsize=(8,4))
    return

def plot_predictions(predicted_values, target_values):

    plt.plot(target_values, label='Actual')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    plt.title("Predicted v. Actual Bike Usage")
    plt.xlabel("Date")
    plt.ylabel("Bike Usage")
    plt.figure(figsize=(8,4))
    return
