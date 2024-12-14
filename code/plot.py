import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from numpy.linalg import norm
import dateutil
import csv
import math



def plot_error():

    print("Plotting error X_err")

    with open('results/best/error.csv', mode='r') as file:
        
        error = list(csv.reader(file))
        error = [[float(x) for x in p] for p in error]
        norm_error = []

        for i in error:
            norm_error.append(norm(i, 1))

        x_err_plot = pd.DataFrame({
                            'time': [ i for i in range(len(norm_error[1:])) ] ,
                            'error': norm_error[1:]
                         })
    
        plt.plot(x_err_plot.time, x_err_plot.error, color='blue')

        plt.title('Evolution of error X_err')
        
        plt.xlabel('Time')
        plt.ylabel('Error')

        plt.show()
