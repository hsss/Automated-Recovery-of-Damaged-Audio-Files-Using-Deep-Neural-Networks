from utils import *

import pickle
import numpy as np

_sequence_length = 170   
_unit_size = 48

_chunk_size = _sequence_length * _unit_size

_src_path = './DB/'
_trg_path = './data/refined_samples_%d_%d'%(_sequence_length, _unit_size)

# load list of all files from the source path
file_list = get_file_list(_src_path)

type2int_dictionary = {}

f_ark = COFF(_trg_path + '.ark', 'wb')
f_scp = open(_trg_path + '.scp', 'w')


for file in file_list:
	print(file)
	tmp = file.split('/')
	
	# make label of each file from directory name
	# ex) file: './DB/'mp3'/audio, label: mp3'
	file_type = tmp[-2]
	
	if file_type not in type2int_dictionary:
		type2int_dictionary[file_type] = len(type2int_dictionary)
	
	# read bits from file, ex) '00101010011110...'
	bits = get_bits_str(file)
		
	# split bits into chunks 
	# size of each chunk is about 1kb
	# shape of each chunk is (170, 6bytes * 8bits) -> (170, 48)
	bit_index = 0
	while bit_index + _chunk_size < len(bits):
	
		x_refined = np.zeros((_sequence_length * _unit_size), dtype = np.int8)
		
		for i in range(_chunk_size):
			if bits[i + bit_index] == '1':
				x_refined[i] = 1
				
		x_refined = x_refined.reshape((_sequence_length, _unit_size))
		f_scp.write('%s_%d %d\n'%(file,bit_index, f_ark.get_pointer()))
		f_ark.save_array(x_refined)
		
		bit_index += _chunk_size
	
pickle.dump(type2int_dictionary, open('data/type2int_dictionary', 'wb'))
	
	








