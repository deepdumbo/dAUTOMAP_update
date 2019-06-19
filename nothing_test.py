import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt

# M =4
# N =2

# complex_type=np.complex64
# inverse=True
# sign = 1 if inverse else -1
# print(sign)
# dft1mat_m = np.zeros((M, M), dtype=complex_type)
# dft1mat_n = np.zeros((N, N), dtype=complex_type)

# for (l, m) in itertools.product(range(M), range(M)):
# 	print(l,m)
# 	dft1mat_m[l,m] = np.exp(sign * 2 * np.pi * 1j * (m * l / M))

# for (k, n) in itertools.product(range(N), range(N)):
#     dft1mat_n[k,n] = np.exp(sign * 2 * np.pi * 1j * (n * k / N))

#     # kronecker product
# mat_kron = np.kron(dft1mat_n, dft1mat_m)
# print(mat_kron.shape)



# def read_mask(mask, mask_name,show_image=None):
#     """
#     read the mask, turn it to list whose value is [0,1]
#     shape is [256 256]
#     """
#     path_test = os.path.join(os.path.expanduser('/'),'home','tuh21221',
#                 'Documents','PythonFile','mask')

#     mask= Image.open(path_test+"/"+"{}".format(mask)+
#                 "/"+"{}.tif".format(mask_name))
#     mask_list = np.asarray(list (mask.getdata() ))

#     mask_list = mask_list / np.amax(mask_list)
#     #either use from future or use // to get float result
#     mask_list = np.reshape(mask_list,(256,256))
#     if (show_image == True):

#         print(mask_list.shape)
#         plt.figure()
#         plt.imshow(mask_list,cmap='gray')
#         plt.show()
#         print(mask_list)
#     return mask_list

# mask_list = read_mask(mask='cartes',mask_name='cartes_10',show_image=True)
# print(mask_list)


# def greet(name):
# 	print('hello,' + name + '!')
# 	greet2(name)
# 	print 'getting ready to say bye...'
# 	bye()


# def greet2(name):
# 	print ('how are you, ' + name + '?')
# def bye():
# 	print 'ok bye!'

# greet(name = 'maggie')

# def sum(list):
# 	if list == []:
# 		return 0 
# 	else: 
# 		print(list[1:])
# 		return list[0] + sum(list[1:])

# a = sum([ 5, 4, 3, 2 ])
# print(a)
 
# b = [ 5, 4, 3, 2 ]
# print(b[1:])s

# def max(list):
# 	 if len(list) == 2:
# 	 	return list[0] if list[0] > list[1] else list[1]
# 	 _max = max(list[1:])
# 	 print (list[1:])
# 	 return list[0] if list[0] > _max else _max

# a = max([5,6,1,2,9,4])
# print (a) 


# from collections import deque
# graph = {}
# graph['you'] = ['alice', 'bob', 'claire'] 

# def search(name):

# 	search_queue = deque() # create a new queue
# 	search_queue += graph[name] # add all your neighbours to the search queue
# 	searched = []

# 	while search_queue: # while the queue isn't empty 
# 		person = search_queue.popleft() # grabs the first person off the quese 
# 		if not person in searched: # only search this guy is you haven't already searched him
# 			if person_is_seller(person):
# 				print person + 'is a mango seller!'
# 				return True 
# 			else: 
# 				search_queue += graph[person] # if they are not, add all of this person's friends to the search queue
# 				searched.append(person)
# 	return False 

# def person_is_seller(name):
# 	return name[-1] == 'm' # check whether the person's name ends with the letter m 