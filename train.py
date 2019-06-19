import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable 
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from generate_input import load_images_from_folder
import models
import os
import shutil
# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#training parameters 
patch_size = 256
EPOCH = 2000
BATCH_SIZE=30
learning_rate=0.0001
mask_path = os.path.join(os.path.expanduser('/'),'home','tuh21221','Documents','PythonFile','mask')

mask_type_cartes='cartes'
mask_name_cartes10='cartes_10'
mask_type_radial='radial'
mask_name_radial10='radial_10'
mask_type_spiral='spiral'
mask_name_spiral10='spiral_10'

torch.manual_seed(1)#reproducible
shutil.rmtree('./checkpoint')
os.mkdir('checkpoint')

# Load training data
tic1 = time.time()
# Folder with images
dir_train = "/data/Catherine/Deep_MRI_Unet/AUTOMAP/test_AUTOMAP/data"
#dir_dev = "/data/Catherine/Deep_MRI_Unet/AUTOMAP/test_AUTOMAP/data"
n_cases = (0,1240) # image number want to load
X_train_cartes, Y_train_cartes = load_images_from_folder( 
	# Load images for training,X_train is undersampled frequency data, 2 channel. Y_train is fully-sampled image
    dir_train,
    n_cases,
    patch_size=patch_size,
    mask_path=mask_path, 
    mask_type=mask_type_cartes, 
    mask_name=mask_name_cartes10,
    normalize=False,
    imrotate=True)

X_train_radial, Y_train_radial = load_images_from_folder( 
	# Load images for training,X_train is undersampled frequency data, 2 channel. Y_train is fully-sampled image
    dir_train,
    n_cases,
    patch_size=patch_size,
    mask_path=mask_path, 
    mask_type=mask_type_radial, 
    mask_name=mask_name_radial10,
    normalize=False,
    imrotate=True)

X_train_spiral, Y_train_spiral = load_images_from_folder( 
	# Load images for training,X_train is undersampled frequency data, 2 channel. Y_train is fully-sampled image
    dir_train,
    n_cases,
    patch_size=patch_size,
    mask_path=mask_path, 
    mask_type=mask_type_spiral, 
    mask_name=mask_name_spiral10,
    normalize=False,
    imrotate=True)

toc1 = time.time()
X_train = np.concatenate((X_train_spiral,X_train_radial,X_train_cartes),axis=0)
Y_train = np.concatenate((Y_train_spiral,Y_train_radial,Y_train_cartes),axis=0)
#transform numpy data to tensor 
X_train_R = X_train[:, :, :, 0]
X_train_R = torch.from_numpy(np.array(X_train_R))
X_train_R = torch.unsqueeze(X_train_R, 1)

X_train_I = X_train[:, :, :, 1]
X_train_I = torch.from_numpy(np.array(X_train_I))
X_train_I = torch.unsqueeze(X_train_I, 1)

Y_train = torch.from_numpy(Y_train).float()
Y_train = torch.unsqueeze(Y_train, 1)
#Y_train = torch.cat((Y_train,Y_train),dim=1)

X_train = torch.cat((X_train_R,X_train_I),dim=1)
X_train,Y_train = Variable(X_train, requires_grad=False),Variable(Y_train, requires_grad=False)
X_train,Y_train = X_train.to(device),Y_train.to(device)


print('Time to load data = ', (toc1 - tic1))
print('X_train.shape at input = ', X_train.shape)
print('Y_train.shape at input = ', Y_train.shape)

#make batch data 
torch_dataset = Data.TensorDataset(X_train, Y_train)
loader = Data.DataLoader(
			dataset = torch_dataset,
			batch_size = BATCH_SIZE,
			shuffle = True
			)

# #training and testing 


recon = models.dAUTOMAPExt(input_shape=(2, patch_size, patch_size), 
                    output_shape=(1, patch_size, patch_size),
                     tfx_params={
                                'nrow': patch_size,
                                'ncol': patch_size,
                                'nch_in': 2,
                                'kernel_size': 1,
                                'nl': 'relu',
                                'init_fourier': False,
                                'init': 'xavier_uniform_',
                                'bias': True,
                                'share_tfxs': False,
                                'learnable': True,}, 
                    depth=2, nl='tanh')


recon.to(device)
# plt.ion() #  plot in real time
# plt.show()
optimizer = torch.optim.Adam(recon.parameters(),lr=learning_rate)
loss_func = torch.nn.MSELoss()



for epoch in range(EPOCH):
	for step, (batch_x, batch_y) in enumerate(loader):
		#prediction = recon(X_train.float())
		prediction = recon(batch_x.float())
		#print(prediction.shape)

		loss = loss_func(prediction,batch_y)
		print('Epoch: ', epoch, '| Step: ', step, '| Loss: ', loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		#cost_mean = np.mean(loss) / BATCH_SIZE
		#learning_curve.append(loss)
	if epoch % 50  ==0:
		#print(loss)
		# plt.cla()
		# plt.scatter(X_train.data.numpy(), Y_train.data.numpy())
		# plt.plot(X_train.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		# plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
		# plt.pause(0.1)
		torch.save(recon, './checkpoint/dAUTOMAP_{}.pkl'.format(epoch))

# plt.plot(learning_curve)
# plt.title('Learning Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Cost')
# plt.show()		
#plt.ioff()
#plt.show()

# def create_placeholders(n_H0, n_W0):
#     """ Creates placeholders for x and y for tf.session
#     :param n_H0: image height
#     :param n_W0: image width
#     :return: x and y - tf placeholders
#     """

#     x = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, 2], name='x')
#     y = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0], name='y')

#     return x, y

# #cost 
# def compute_cost(DECONV, Y):
#     """
#     Computes cost (squared loss) between the output of forward propagation and
#     the label image
#     :param DECONV: output of forward propagation
#     :param Y: label image
#     :return: cost (squared loss)
#     """

#     cost = tf.square(DECONV - Y)

#     return cost

# #model 
# def model(X_train, Y_train, learning_rate=0.0001,
#           num_epochs=100, minibatch_size=5, print_cost=True):
#     """ Runs the forward and backward propagation
#     :param X_train: input training frequency-space data
#     :param Y_train: input training image-space data
#     :param learning_rate: learning rate of gradient descent
#     :param num_epochs: number of epochs
#     :param minibatch_size: size of mini-batch
#     :param print_cost: if True - the cost will be printed every epoch, as well
#     as how long it took to run the epoch
#     :return: this function saves the model to a file. The model can then
#     be used to reconstruct the image from frequency space
#     """

#     with tf.device('/gpu:0'):
#         ops.reset_default_graph()  # to not overwrite tf variables
#         seed = 3
#         (m, n_H0, n_W0, _) = X_train.shape

#         # Create Placeholders
#         X, Y = create_placeholders(n_H0, n_W0)

#         # Initialize parameters
#         parameters = initialize_parameters()

#         # Build the forward propagation in the tf graph
#         DECONV = forward_propagation(X, parameters)

#         # Add cost function to tf graph
#         cost = compute_cost(DECONV, Y)
#         #tf.print(DECONV)
#         #tf.Print(Y)
# #        # Backpropagation
# #        optimizer = tf.train.RMSPropOptimizer(learning_rate,
# #                                              decay=0.9,
# #                                              momentum=0.0).minimize(cost)
        
#         # Backpropagation
#         # Add global_step variable for save training models - Chong Duan
#         my_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
#         # optimizer = tf.train.RMSPropOptimizer(learning_rate,
#         #                                       decay=0.9,
#         #                                       momentum=0.0).minimize(cost, global_step = my_global_step)
#         optimizer = tf.train.AdamOptimizer(learning_rate,
#                                               beta1=0.5).minimize(cost, global_step = my_global_step)

#         # Initialize all the variables globally
#         init = tf.global_variables_initializer()

#         # Add ops to save and restore all the variables
#         saver = tf.train.Saver(save_relative_paths=True)

#         # For memory
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True

#         # Memory config
#         #config = tf.ConfigProto()
#         #config.gpu_options.allow_growth = True
#         config = tf.ConfigProto(log_device_placement=True)

#         # Start the session to compute the tf graph
#         with tf.Session(config=config) as sess:

#             # Initialization
#             sess.run(init)

#             # Training loop
#             learning_curve = []
#             for epoch in range(num_epochs):
#                 tic = time.time()

#                 minibatch_cost = 0.
#                 num_minibatches = int(m / minibatch_size)  # number of minibatches
#                 seed += 1
#                 minibatches = random_mini_batches(X_train, Y_train,
#                                                   minibatch_size, seed)
#                 # Minibatch loop
#                 for minibatch in minibatches:
#                     # Select a minibatch
#                     (minibatch_X, minibatch_Y) = minibatch
#                     # Run the session to execute the optimizer and the cost
#                     _, temp_cost = sess.run(
#                         [optimizer, cost],
#                         feed_dict={X: minibatch_X, Y: minibatch_Y})

#                     cost_mean = np.mean(temp_cost) / num_minibatches
#                     minibatch_cost += cost_mean

#                 # Print the cost every epoch
#                 learning_curve.append(minibatch_cost)
#                 if print_cost:
#                     toc = time.time()
#                     print ('EPOCH = ', epoch, 'COST = ', minibatch_cost, 'Elapsed time = ', (toc - tic))
                    
#                 if (epoch + 1) % 100 == 0:
#                     save_path = saver.save(sess, './checkpoints/model.ckpt', global_step = my_global_step)
#                     print("Model saved in file: %s" % save_path)


# #            # Save the variables to disk.
# #            save_path = saver.save(sess, './model/' + 'model.ckpt')
# #            print("Model saved in file: %s" % save_path)
            
#             # Plot learning curve
#             plt.plot(learning_curve)
#             plt.title('Learning Curve')
#             plt.xlabel('Epoch')
#             plt.ylabel('Cost')
#             plt.show()
            
#             # Close sess
#             sess.close()

# # Finally run the model!
# model(X_train, Y_train,
# #      learning_rate=0.00002,
#       learning_rate=0.0001,
#       num_epochs=2000,
#       minibatch_size=66,  # should be < than the number of input examples
#       print_cost=True)
