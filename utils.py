from os import walk
from os.path import splitext
import numpy as np
import gzip

class COFF():
	def __init__(self, fn, mode, compress = False):
		if compress:
			self.f = gzip.open(fn, mode, compresslevel=1)
		else:
			self.f = open(fn, mode)
	def get_pointer(self):
		return self.f.tell()
		
	def set_pointer(self, pointer):
		self.f.seek(pointer)
		
	def save_array(self, data):
		np.save(self.f, data)
		
	def load_array(self):
		return np.load(self.f)
		
	def load_array_at(self, pointer):
		self.f.seek(pointer)
		return np.load(self.f)
		
	def close(self):
		self.f.close()
		
def get_file_list(dirname):
	
	file_list = []

	for (path, dir, files) in walk(dirname):
		for filename in files:
			file_list.append(path + '/' + filename)

	return file_list
	
def get_bits_str(file):
	f = open(file, 'rb')
	data_block = f.read()
	f.close()
	
	bytes = (b for b in data_block)
	bits = ''
	
	for b in bytes:
		for j in reversed((range(8))):
			bits += str((b >>j) & 1)
			
	return bits