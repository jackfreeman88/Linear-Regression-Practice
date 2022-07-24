'''data.py
Reads CSV files, stores data, access/filter data by variable name
Jack Freeman
CS 251 Data Analysis and Visualization
Spring 2021
'''
import sys
import numpy as np
import csv


class Data:

	#constructor	
	def __init__(self, filepath=None, headers=None, data=None, header2col=None):	

		self.filepath = filepath
		self.headers = headers
		self.data = data
		self.header2col = header2col
		
		if (self.filepath != None):
			self.read(self.filepath)

	#reads in .csv file and stores data in a numpy ndarray
	def read(self, filepath):
		
		self.filepath = filepath
		self.headers = []
		self.header2col = {}
		self.data = []
		types = []
		non_num_indx = []
		indx = 0
		
		with open(filepath, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			self.headers = next(csvreader)
			types = next(csvreader)
			
			if ('numeric' not in types):
				print("Error: Non numeric")
				sys.exit()
			for val in types:
				val = val.strip()
				if (val != 'numeric'):
					non_num_indx.append(indx)
				indx += 1
				
			rad = 0
			for val in non_num_indx:
				self.headers.pop(val-rad)
				rad += 1
			for i in range(len(self.headers)):
				self.headers[i] = self.headers[i].strip()
				if (type(self.headers[i]) != str):
					print("Invalid Headers")
					sys.exit()
				self.header2col[self.headers[i]] = i
				
			for row in csvreader:
				rad = 0
				for val in non_num_indx:
					row.pop(val-rad)
					rad += 1
				self.data.append([float(i) for i in row])
			
		self.data = np.array(self.data)
		
#     toString method
	def __str__(self):

		if (len(self.data) < 5):
			numRows = len(self.data)
		else:
			numRows = 5
		print('---------------------')
		print(f'{self.filepath} ({len(self.data)}x{len(self.headers)})')
		print('Headers:')
		print(' ' + ' '.join(self.headers))
		print('---------------------')
		print(f'Showing first {numRows}/{len(self.data)} rows.')
		for i in range(numRows):			
			print(' '.join(map(str,self.data[i])))
		print('---------------------')
		
		return "" 
	
	def get_headers(self):
		'''Get method for headers'''
		return self.headers

	
	def get_mappings(self):
		'''Get method for mapping between variable name and column index'''
		return self.header2col

	
	def get_num_dims(self):
		'''Get method for number of dimensions in each data sample'''
		return len(self.headers)

	
	def get_num_samples(self):
		'''Get method for number of data points (samples) in the dataset'''
		return len(self.data)

	
	def get_sample(self, rowInd):
		'''Gets the data sample at index `rowInd` (the `rowInd`-th sample)'''
		return self.data[rowInd]

	
	def get_header_indices(self, headers):
		'''Gets the variable (column) indices of the str variable names in `headers`.'''
		ind = []
		for val in headers:
			ind.append(self.header2col[val])
		return ind
				

	
	def get_all_data(self):
		'''Gets a copy of the entire dataset'''
		return(self.data.copy())

	
	
	
	def head(self):
		'''Return the 1st five data samples (all variables)'''
		return(self.data[:5])

	
	
	
	def tail(self):
		'''Return the last five data samples (all variables)'''
		return(self.data[-5:])

	
	
	def limit_samples(self, start_row, end_row):
		'''Update the data so that this `Data` object only stores samples in the contiguous range:
			`start_row` (inclusive), end_row (exclusive)
		Samples outside the specified range are no longer stored.

		(Week 2)

		'''
		self.data = self.data[start_row:end_row]

	def select_data(self, headers, rows=[]):
		'''Return data samples corresponding to the variable names in `headers`.
		If `rows` is empty, return all samples, otherwise return samples at the indices specified
		by the `rows` list.'''
		
		if type(headers) != list:
			headers = [headers]
			
		cols = self.get_header_indices(headers)
		if (rows == []):
			return(self.data[np.ix_(np.arange(self.data.shape[0]),cols)])
		else:
			return(self.data[np.ix_(rows,cols)])		
	
