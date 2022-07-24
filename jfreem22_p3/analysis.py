'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Jack Freeman
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt

#Constructor
class Analysis:
    def __init__(self, data):

        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})


#Setter
    def set_data(self, data):

        self.data = data
#Computes min of each var in headers
    def min(self, headers, rows=[]):

        if (rows == []):
            return np.min(self.data.select_data(headers), axis=0)
        else:
            return np.min(self.data.select_data(headers, rows), axis=0)
        
#Computes max of each var in headers
    def max(self, headers, rows=[]):

        if (rows == []):
            return np.max(self.data.select_data(headers), axis=0)
        else:
            return np.max(self.data.select_data(headers, rows), axis=0)


#Computes range of each var in headers
    def range(self, headers, rows=[]):

        if (rows == []): 
            return [np.min(self.data.select_data(headers), axis=0), np.max(self.data.select_data(headers), axis=0)]
        else:
            return [np.min(self.data.select_data(headers, rows), axis=0), np.max(self.data.select_data(headers, rows), axis=0)]


#Computes mean of each var in headers
    def mean(self, headers, rows=[]):

        if (rows == []):
            return np.sum(self.data.select_data(headers), axis=0)/self.data.get_num_samples()
        else:
            return np.sum(self.data.select_data(headers, rows), axis=0)/len(rows)


#Computes var of each var in headers
    def var(self, headers, rows=[]):

        if (rows == []):
            return np.sum(np.square(self.data.select_data(headers)-self.mean(headers)), axis=0)/(self.data.get_num_samples()-1)
        else:
            return np.sum(np.square(self.data.select_data(headers, rows)-self.mean(headers,rows)), axis=0)/(len(rows)-1)



#Computes std of each var in headers
    def std(self, headers, rows=[]):

        return np.sqrt(self.var(headers, rows))

    def show(self):

        plt.show()

#Create scatter plot
    def scatter(self, ind_var, dep_var, title):


        x = self.data.select_data([ind_var])
        y = self.data.select_data([dep_var])
        x = np.squeeze(x)
        y = np.squeeze(y)

        plt.scatter(x,y)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        
        
        return x,y


#Creates a pair plot       
    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        fig, axes = plt.subplots(len(data_vars),len(data_vars), figsize=fig_sz)
        fig.suptitle(title)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                axes[i,j].scatter(self.data.select_data(data_vars[j]), self.data.select_data(data_vars[i]))
                axes[i,j].set(xlabel=data_vars[j], ylabel=data_vars[i])
                axes[i,j].label_outer()
        
        return fig, axes