from keras.layers import LSTM, Input, Dropout, Dense
from keras.models import Model, model_from_json
from threading import Thread
from time import sleep
from utils import *

import numpy as np

import queue
import pickle
global q
q = queue.Queue(maxsize=1000)


_num_layers = 2
_num_nodes = 20

_num_epoch = 20
_batch_size = 500
_tr_scp = 'data/refined_samples_170_48.scp'
_tr_ark = 'data/refined_samples_170_48.ark'

# make mini-batch (including x,y), and pass it using queue
def make_batch(tr_lines, f_ark, batch_size, num_epoch, type2int_dictionary):
	for i in range(num_epoch):
		np.random.shuffle(tr_lines)
		
		for j in range(num_batchs):
			x = []
			y = []
			for line in tr_lines[j: j+batch_size]:
				tmp = line.strip().split(' ')
				fn = tmp[0]
				file_type = fn.split('/')[-2]
				y.append(type2int_dictionary[file_type])
				
				pointer = int(tmp[1])
				x.append(f_ark.load_array_at(pointer))
				
			x = np.array(x, np.float32)
			y = np.array(y, np.float32)
			
			q.put([x, y])
			
	

type2int_dictionary = pickle.load(open('data/type2int_dictionary','rb'))

# LSTM-based model construction
main_input = Input(shape=(170, 48),name='main_input')
l = main_input

for i in range(_num_layers):
	if i == _num_layers - 1:
		return_sequences = False
	else:
		return_sequences = True		
	
	l = LSTM(_num_nodes, dropout=0.5, recurrent_dropout=0.1, return_sequences=return_sequences, stateful=False)(l)
	
l = Dropout(0.5)(l)
result = Dense(len(type2int_dictionary), activation='softmax', name = 'result')(l)

model = Model(inputs = [main_input], outputs = [result])

# compile model using rmspropagation optimizer
model.compile(	loss={'result': 'sparse_categorical_crossentropy'},
			    optimizer='rmsprop', metrics=['accuracy'])

tr_lines = open(_tr_scp, 'r').readlines()
num_batchs = int(len(tr_lines) / _batch_size)
p = Thread(target = make_batch, args = (tr_lines, COFF(_tr_ark, 'rb'), _batch_size, _num_epoch, type2int_dictionary))
p.start()
	
for i in range(_num_epoch):
	loss_list = []
	acc_list = []
	for j in range(num_batchs):
		while True:
			if q.empty():
				sleep(0.1)

			else:
				x, y = q.get()
				loss, acc = model.train_on_batch(x, y)
				#print(loss, acc)
				
				loss_list.append(loss)
				acc_list.append(acc)
				break
	print('epoch:%d loss:%f acc:%f'%(i, np.mean(loss_list), np.mean(acc_list)))
	
# save the model after training
json_string = model.to_json()
open('data/LSTM_%dlayer_%dnode.json'%(_num_layers, _num_nodes), 'w').write(json_string)
model.save_weights('data/LSTM_%dlayer_%dnode.h5'%(_num_layers, _num_nodes))
	
	
	