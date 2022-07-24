'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Jack Freeman
CS251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
	'''
	Perform and store linear regression and related analyses
	'''

	def __init__(self, data):
		'''

		Parameters:
		-----------
		data: Data object. Contains all data samples and variables in a dataset.
		'''
		super().__init__(data)

		# ind_vars: Python list of strings.
		#	1+ Independent variables (predictors) entered in the regression.
		self.ind_vars = None
		# dep_var: string. Dependent variable predicted by the regression.
		self.dep_var = None

		# A: ndarray. shape=(num_data_samps, num_ind_vars)
		#	Matrix for independent (predictor) variables in linear regression
		self.A = None

		# y: ndarray. shape=(num_data_samps, 1)
		#	Vector for dependent variable predictions from linear regression
		self.y = None

		# R2: float. R^2 statistic
		self.R2 = None

		# Mean SEE. float. Measure of quality of fit
		self.m_sse = None

		# slope: ndarray. shape=(num_ind_vars, 1)
		#	Regression slope(s)
		self.slope = None
		# intercept: float. Regression intercept
		self.intercept = None
		# residuals: ndarray. shape=(num_data_samps, 1)
		#	Residuals from regression fit
		self.residuals = None

		# p: int. Polynomial degree of regression model (Week 2)
		self.p = 1

	def linear_regression(self, ind_vars, dep_var):
		'''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
		and dependent variable `dep_var. '''
	   
	   
		self.ind_vars = ind_vars
		self.dep_var = dep_var
		ind_data = self.data.select_data(ind_vars)
		dep_data = self.data.select_data(dep_var)
		dep = np.array(dep_data)
		A = np.array(ind_data)
		self.A = A
		Ahat = np.hstack((np.ones((A.shape[0],1)), A))
		c, _, _, _ = scipy.linalg.lstsq(Ahat, dep)
		
		self.y = Ahat @ c
		self.intercept = float(c[0])
		self.slope = c[1:,]
		self.R2 = self.r_squared(self.y)
		self.residuals = self.compute_residuals(self.y)

		
		

	def predict(self, X=None):
		'''Use fitted linear regression model to predict the values of data matrix self.A.
		Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
		A is the data matrix. '''
		
		if X is None:
			y_pred = np.matmul(self.A, self.slope)
		else:
			y_pred = np.matmul(X, self.slope)
		y_pred += self.intercept
		return y_pred

	def r_squared(self, y_pred):
		'''Computes the R^2 quality of fit statistic '''
	   
		
		y_data = self.data.select_data(self.dep_var)
		mean = self.mean(self.dep_var)
		sq = ((y_data- mean)**2)
		ss = np.sum(sq)
		rss = np.sum((y_data - y_pred)**2)
		R2 = 1 - (rss/ss)
		
		return R2

	def compute_residuals(self, y_pred):
		'''Determines the residual values from the linear regression model '''
	   
		y_data = self.data.select_data(self.dep_var)
		res = y_data - y_pred
		return res

	def mean_sse(self):
		'''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
		See notebook for equation.'''
	   
		msse = np.mean(self.residuals**2)
		return msse

	def scatter(self, ind_var, dep_var, title):
		'''Creates a scatter plot with a regression line to visualize the model fit.
		Assumes linear regression has been already run. '''
		
		
		msse = self.mean_sse()
		new_title = title + f" R2 is {self.R2:.4f}"
		x, y = super().scatter(ind_var, dep_var,  title = new_title)
		xline = np.linspace(np.min(x), np.max(x), len(x))
	   
		
		if self.p == 1:
			y_pred = xline * self.slope[0] + self.intercept
		
		else:
			xlinepoly = self.make_polynomial_matrix(xline, self.p) 
			datapoly = np.sum((xlinepoly @ self.slope), axis = 1)
			y_pred = self.intercept + datapoly

		plt.xlabel(ind_var)
		plt.ylabel(dep_var)
		plt.plot(xline,y_pred,'-r')
		
	def pair_plot(self,data_vars,fig_sz=(12, 12),hists_on_diag=True):
		'''Makes a pair plot with regression lines in each panel.
		There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
		on x and y axes.'''
		fig, axs = super().pair_plot(data_vars)
		for i in range(len(data_vars)):
			for j in range(len(data_vars)):
				if (i == j and hists_on_diag == True):
					numVars = len(data_vars)
					axs[i, j].remove()
					axs[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
					if j < numVars-1:
						axs[i, j].set_xticks([])
					else:
						axs[i, j].set_xlabel(data_vars[i])
					if i > 0:
						axs[i, j].set_yticks([])
					else:
						axs[i, j].set_ylabel(data_vars[i])
					axs[i,j].hist(self.data.select_data(data_vars[j]))
					axs[i,j].title.set_text(f'Hist of {data_vars[j]}')
				else:
					self.linear_regression(data_vars[j], data_vars[i])
					line_x = np.linspace(self.min(data_vars[j]),self.max(data_vars[j]),num = self.A.shape[0]*10)
					line_y = self.slope*line_x + self.intercept
					axs[i,j].plot(line_x,line_y,color='r')
					axs[i,j].title.set_text(f'R^2: {self.R2:.2f}')


		
		
		
		
		
		
				

##################################################################################################################################

	def make_polynomial_matrix(self, A, p):
		'''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
		for a polynomial regression model of degree `p`. '''
	   
		poly_matrix = np.zeros((A.shape[0],p))
		for i in range(1,p+1):
			new_col = A.copy()
			poly_matrix[:,i-1] = np.squeeze(np.power(new_col,i))
		
		return poly_matrix
            

	def poly_regression(self, ind_var, dep_var, p):
		'''Perform polynomial regression — generalizes self.linear_regression to polynomial curves '''
		
	
		self.ind_vars = ind_var
		self.dep_var = dep_var
		self.p = p
		self.A = self.make_polynomial_matrix(self.data.select_data(self.ind_vars), p)
		self.y = self.data.select_data([dep_var])

		
		reg = np.hstack((self.A, np.ones((self.A.shape[0],1)))) 
		
		
		c, _, _, _ = scipy.linalg.lstsq(reg, self.y) 
		
	
		self.slope = c[:-1]
		self.intercept = c[-1][0]
		y_pred = self.predict()
		self.R2 = self.r_squared(y_pred)
		self.residuals = self.compute_residuals(y_pred)
		self.m_sse = self.msse()

	def get_fitted_slope(self):
		'''Returns the fitted regression slope.'''
		return self.slope

	def get_fitted_intercept(self):
		'''Returns the fitted regression intercept.'''
		return self.intercept

	def initialize(self, ind_vars, dep_var, slope, intercept, p):
		'''Sets fields based on parameter values.'''
		
		self.ind_vars = ind_vars
		self.dep_var = dep_var
		self.slope = slope
		self.intercept = intercept
		self.p = p 

		
		if self.p == 1:
			self.A = self.data.select_data([ind_vars])
		else:
			self.A = self.make_polynomial_matrix(self.data.select_data([ind_vars]), p)

		self.y = self.data.select_data([dep_var])
		y_pred = self.predict()
		self.residuals = self.compute_residuals(y_pred)
		self.R2 = self.r_squared(y_pred)
		self.m_sse = self.msse()


