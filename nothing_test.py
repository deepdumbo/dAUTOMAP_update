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

import collections
class TrieNode:
	def __init__(self):
		self.children = collections.defaultdict(TrieNode)
		self.is_word = False 

class Trie:
	def __init__(self):
		self.root = TrieNode()

	def insert(self,word):
		current = self.root
		for letter in word:
			current = current.children[letter]
			print('current',current)
		current.is_word = True 

	def search(self,word):
		node = self.root
		self.res = False 
		self.dfs(node,word)
		return self.res

	def dfs(self,node,word):
		print('word', word)
		print('node', node)
		if not word:
			print('word', word)
			if node.is_word:
				self.res = True 
			return 
		if word[0] == '.':
			for n in node.children.values():
				print(' node.children.values()', node.children.values())
				self.dfs(n,word[1:])
		else:
			node = node.children.get(word[0])
			if not node:
				return
			self.dfs(node,word[1:])



		# for letter in word:
		# 	current = current.children.get(letter)
		# 	if current is None:
		# 		return False 
		# return current.is_word

	# def startsWith(self,prefix):
	# 	current = self.root
	# 	for letter in prefix:
	# 		current = current.children.get(letter)
	# 		if current is None:
	# 			return False 
	# 	return True 

obj = Trie()
obj.insert('bad')
obj.insert('dad')
obj.insert('mad')
obj.insert('pad')

obj.search('bad')
obj.search('.ad')
obj.search('b..')



# param_2 = obj.search('apple')
# param_3 = obj.startsWith('app')
# print (param_2,param_3)
