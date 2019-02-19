import os
import pickle
import numpy as np

from keras.models import model_from_json
from utils import *

_sequence_length = 170
_unit_size = 48

_chunk_size = _sequence_length * _unit_size


def get_docoding_result(case, model):
	bits = get_bits_str(case)

	chunk_list = []

	bit_index = 0
		
	while bit_index + _chunk_size < len(bits):
		x_refined = np.zeros((_sequence_length * _unit_size), dtype = np.int8)
		
		for i in range(_chunk_size):
			if bits[i + bit_index] == '1':
				x_refined[i] = 1
				
		x_refined = x_refined.reshape((_sequence_length, _unit_size))
		chunk_list.append(x_refined)
		bit_index += _chunk_size
		
	chunk_list = np.array(chunk_list, np.float32)
	result = model.predict(chunk_list)
	
	return result

# load model 
model = model_from_json(open('data/LSTM_2layer_20node.json').read())
model.load_weights('data/LSTM_2layer_20node.h5')
type2int_dictionary = pickle.load(open('data/type2int_dictionary','rb'))
int2type_dictionary = {}

for type in type2int_dictionary:
	int2type_dictionary[type2int_dictionary[type]] = type

# confirm recoding results of unseen files
for case in ['cases/QtxuI4G-emg_0000004.mp3', 'cases/QtxuI4G-emg_0000004.wav']:
	result = get_docoding_result(case, model)

	majority_voting = np.sum(result, axis = 0)
	print('docding result from %s:'%(case), int2type_dictionary[np.argmax(majority_voting)])























