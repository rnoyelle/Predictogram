 
import sys
import os
import os.path

import lib.cnn.matnpyio as io
import lib.cnn.cnn as cnn 
import lib.cnn.matnpy as matnpy
import lib.cnn.preprocess as pp

import tensorflow as tf
import numpy as np
from math import ceil

import random
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold

import pandas as pd
import datetime


################################################
#################### PARAMS ####################
################################################

#################
### base path ###
#################

base_path = '/media/rudy/disk2/lucy/'

###################
### data params ###
###################

sess_no ='150128'

# path
raw_path = base_path +sess_no+'/session01/'
rinfo_path = base_path +sess_no+'/session01/' + 'recording_info.mat'
tinfo_path = base_path +sess_no+'/session01/' + 'trial_info.mat'

only_correct_trials = True

#align_on, from_time, to_time = 'sample', 0, 500 
#lowcut, highcut, order = 30, 300, 3



lowcut1, highcut1, order1 = 8, 14, 3 
lowcut2, highcut2, order2 = lowcut1, highcut1, order1

window_size1 = 200
window_size2 = window_size1 #window_size1 ### 0.72
step = 100
#delay = 0

eps = 100  # start at 0 + eps and finish at end - eps (prevent zero padding for frequency filters)
# security margin for zero padding cominf by the frequency filter

renorm = True

select_elec_by = 'cortex' # 'areas' or 'cortex'

if select_elec_by == 'areas':
    areas1 = ['a9/46D']
    areas2 = ['a8M'] 
    cortex1='Prefontal'
    cortex2='Prefontal'
elif select_elec_by == 'cortex':
    cortex1 = 'Visual' # coding <U16
    cortex2 = 'Visual' # 

    areas1 = io.get_area_cortex(rinfo_path, cortex1, unique = True)
    areas2 = io.get_area_cortex(rinfo_path, cortex2, unique = True)
    
    
only_correct_trials = False
only_align_on_available = False

    
    
#######################
### LEARNING PARAMS ###
#######################

train_size = 0.8
test_size = 1 - train_size
seed = np.random.randint(1,10000)

size_of_batches = 1 # 50
n_iterations = 50000 #100
n_epochs = 1 #1  ##apprendre le même lot/batch plusieurs fois avant de passer au suivant


#################
### LOAD DATA ###
#################

data1_all = matnpy.get_subset(sess_no, raw_path, 
                              lowcut1, highcut1,
                                   order = 3,
                                   only_correct_trials=only_correct_trials, only_align_on_available=only_align_on_available, renorm = True )


# data1_all, data2_all = matnpy.get_subset2(sess_no, raw_path, 
#                                    lowcut1, highcut1,
#                                    lowcut2, highcut2,
#                                    step, delay,
#                                    order = 3,
#                                    only_correct_trials = False, only_align_on_available=False, renorm = True )



area_names1 = io.get_area_names(rinfo_path)
area_names2 = area_names1.copy()

sample_on = io.get_sample_on(tinfo_path)
match_on = io.get_match_on(tinfo_path)


######################
### NETWORK PARAMS ###
######################

n_layers = 3  ## 7 for stim, 6 for resp  ###" entre 20 000 et 30 000 params
in1, out1 = 1, 6
in2, out2 = 6, 9
in3, out3 = 9, 12

in4, out4 = 8, 16
in5, out5 = 16, 32
in6, out6 = 32, 64
in7, out7 = 64, 128
in8, out8 = 128, 256



channels_in  = [in1, in2, in3, in4, in5, in6, in7][:n_layers]
channels_out = [out1, out2, out3, out4, out5, out6, out7][:n_layers]

patch_dim = [1, 5]
pool_dim = [1, 2]
patch_height, patch_width = pool_dim[0], pool_dim[1]


learning_rate = 1e-4

weights_dist = 'random_normal'
normalized_weights = True

fc_conv = 50 ##  FC1 : dim1 to fc_conv + nonlin
fc_central = 20 ## FC2 : fc_conv to fc_central
fc_deconv = 50 ## FC3 : fc_central to fc_deconv + nonlin
##  + FC4 : fc_deconv to dim2 + nonlin

kpt = True ## dropout only on FC1 (small flatten to big one (fc2))
keep_prob_train = 0.5 # 0.5 for stim, 0.1 for resp 
if kpt == False:
    keep_prob_train = 1.0

bn = False # batch_norm
DECAY = 0.9 # 0.9 , 0.99, 0.999

l2 = False
l2_regularization_penalty = 0
if l2 == False:
    l2_regularization_penalty = 0
    
    
    
#classes = 5
n_chans1 = 1#data1.shape[1]
samples_per_trial1 = window_size1 #data1.shape[2]

n_chans2 = 1 #data2.shape[1]
samples_per_trial2 = window_size2 #data2.shape[2]


################################################
#                  CREATE CNN                  #
################################################

###################
### PLACEHOLDER ###
###################

x_ = tf.placeholder(tf.float32, shape=[
        None, n_chans1, samples_per_trial1, 1
])
y_ = tf.placeholder(tf.float32, shape=[
        None, n_chans2 , samples_per_trial2, 1
])
#x_stim = tf.placeholder(tf.float32, shape=[
    #None, classes
#])

keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder_with_default(True, shape=())

###################
##### LAYERS  #####
###################

"""
n_layers of convolution layers
4 or 3 ? fully connected layers
n_layers of deconvolution layers (the last one without activation function/nonlin function)
"""

out, weights = cnn.create_network(x_,
                                n_layers,
                                channels_in,
                                channels_out,
                                fc_conv, 
                                fc_central,
                                fc_deconv,
                                patch_dim,
                                pool_dim,
                                training,
                                keep_prob,
                                n_chans2,
                                samples_per_trial2,                   
                                weights_dist='random_normal',
                                normalized_weights = True,
                                nonlin = 'leaky_relu',
                                bn =  bn,
                                kpt = kpt, 
                                DECAY = DECAY)

############
### LOSS ###
############                print('var - mse :', var_test - mse_test)

#mse = tf.reduce_mean( tf.square(out - y_))

if l2 == False:
    loss = tf.reduce_mean( tf.square(out - y_)) # MSE
else :
    loss = cnn.l2_loss(weights, l2_regularization_penalty, out, y_, 'loss') # MSE + l2 penalty
    
#################
### OPTIMIZER ###
#################
    
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


for area1 in areas1 :
    idx1 = []
    for count, area in enumerate(area_names1):
        if area == area1 :
            idx1.append(count)
            
    for count1, idx_channel1 in enumerate(idx1):
        
        for area2 in areas2 :
            idx2 = []
            for count, area in enumerate(area_names2):
                if area == area2 :
                    idx2.append(count)
                    
            for count2, idx_channel2 in enumerate(idx2):
                
                print(area1, count1, area2, count2)
                
                
                data1 = [data1_all[i][[idx_channel1],:] for i in range(len(data1_all))]
                data2 = [data1_all[i][[idx_channel2],:] for i in range(len(data1_all))]
                
                ################################################
                #         TRAINING AND TEST NETWORK            #
                ################################################

                indices = [i for i in range(len(data1))]


                #targets = [ (time_step[i]/step) * classes + np.argmax(stim[i]) for i in range(data1.shape[0]) ]

                x_train, x_test, y_train, y_test, ind_train, ind_test = (
                    train_test_split(
                        data1, 
                        data2, 
                        indices,
                        test_size=test_size, 
                        random_state=seed
                #         stratify = targets
                        )
                    ) 
                
                
                #list_iteration = []
                #list_train = []
                #list_test = []
                


                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    
                    # BASE TEST    
                    curr_x_test = []
                    curr_y_test = []
                    time_step = []
                    for n_trial in ind_test :
                    #n_time = data1[n_trial].shape[1]
                        for n_step in range(int((data1[n_trial].shape[1] - 2*eps - window_size1)/step)+1):
                            curr_x_test.append( data1[n_trial][:, eps + n_step *step : eps + n_step * step + window_size1 ])
                            curr_y_test.append( data2[n_trial][:, eps + n_step *step : eps + n_step * step + window_size2 ])
                            
                            if eps + n_step *step + window_size1/2 < match_on[n_trial].item() :
                                time_step.append(eps + n_step *step + window_size1/2 - sample_on[n_trial].item())
                            else:
                                time_step.append(eps + n_step *step + window_size1/2 - match_on[n_trial].item() + 3000)
                            
                            
                    curr_x_test = np.array(curr_x_test)[:,:,:,None]
                    curr_y_test = np.array(curr_y_test)[:,:,:,None]
                    
                    var_test = np.var(curr_y_test)
                    
                    # TRAINING
                    for i in range(int(n_iterations)):
                        
                        
                        random.shuffle(ind_train)
                        ind_train_batch = ind_train[:size_of_batches]

                #         curr_x = data1[ind_train_batch]
                #         curr_y = data2[ind_train_batch]


                        # TRAIN BASE
                        curr_x = []
                        curr_y = []
                        for n_trial in ind_train_batch :
                            #n_time = data1[n_trial].shape[1]
                            for n_step in range(int((data1[n_trial].shape[1] - 2*eps - window_size1)/step)+1):
                                curr_x.append( data1[n_trial][:, eps + n_step *step : eps + n_step * step + window_size1 ])
                                curr_y.append( data2[n_trial][:, eps + n_step *step : eps + n_step * step + window_size2 ])
                                

                                
                        curr_x = np.array(curr_x)[:,:,:,None] # dim = batch, n_chans, time, 1
                        curr_y = np.array(curr_y)[:,:,:,None] # add a dim = 1
                        
                        
                        for j in range(n_epochs):

                            # TRAIN NETWORK
                            train_step.run(feed_dict={
                                x_ : curr_x,                
                                y_ : curr_y,
                                keep_prob : keep_prob_train
                                })

                            ## PRINT LOSS ON THE CURRENT BATCH
                            #if i%100 ==0:

                                #mse_train = loss.eval(feed_dict={
                                    #x_ : curr_x,
                                    #y_ : curr_y,
                                    #keep_prob : 1.0

                                #})


                                #mse_test = loss.eval(feed_dict={
                                        #x_ : curr_x_test,
                                        #y_ : curr_y_test,
                                        #keep_prob : 1.0
                                    #})

                                #var_train = np.var(curr_y)

                                #print('step %d.%d ; LOSS : %g ; R_train : %g, R_test : %g  '  % (
                                        #i, j, mse_train, 1-mse_train/var_train, 1-mse_test/var_test) )
                                
                                ##list_iteration.append(i)
                                ##list_train.append(mse_train)
                                ##list_test.append(mse_test)
                                
                                
                    mse_train = loss.eval(feed_dict={
                            x_ : curr_x,
                            y_ : curr_y,
                            keep_prob : 1.0

                        })
                        


                    # MSE BASE TEST 

                    mse_test = loss.eval(feed_dict={
                        x_ : curr_x_test,
                        y_ : curr_y_test,
                        keep_prob : 1.0
                    })

                    print('MSE TEST : %g' %(mse_test))
                    #print('R2 TEST : %g' % (1-mse_test/var_test))

                    
                    # ERROR BAR OF R2
                    
                    y_predict =  out.eval(feed_dict={
                        x_ : curr_x_test,
                        y_ : curr_y_test,
                        keep_prob : 1.0
                    })
                    
                    # n_trial, n_chans, n_time, 1
                    
                    mse_test_batch = np.mean( (y_predict - curr_y_test) **2, axis = (1,2,3) )
                    r2_error_bar = np.std(mse_test_batch)/(np.sqrt(mse_test_batch.shape[0])* np.var(curr_y_test) )

                    
                    print('R2 TEST : %g +- %g' % (1-mse_test/var_test, r2_error_bar))
                    
                    
                    # time
                    
                    r2_test_batch = 1 - mse_test_batch/np.var(curr_y_test, axis=(1,2,3) )
                    
                    nonlin = 'leaky_relu'
                    str_freq1 = 'low'+str(lowcut1)+'high'+str(highcut1)+'order'+str(order1)
                    str_freq2 = 'low'+str(lowcut2)+'high'+str(highcut2)+'order'+str(order2)
                    
                    data_size = 0
                    for i in range(len(data1)):
                        data_size += data1[i].shape[1]
                    
                    

                    data_tuning = [ sess_no, area1, count1, area2, count2,   
                                    cortex1, cortex2,    
                                    str_freq1, str_freq2,  
                                    window_size1, window_size2, 
                                    step,# delay,  
                                    len(data1), data_size, 
                                    n_chans1, n_chans2, 
                                    only_correct_trials, only_align_on_available,
                                    mse_test, mse_train, var_test,  
                                    1 -mse_test/var_test, r2_error_bar, var_test - mse_test,  r2_test_batch, time_step,
                                    n_iterations, n_epochs, size_of_batches,
                                    learning_rate,
                                    l2, l2_regularization_penalty, 
                                    kpt, keep_prob_train,
                                    bn, DECAY,
                                    str(patch_dim), str(pool_dim), nonlin,  
                                    fc_conv, fc_central, fc_deconv, 
                                    n_layers, 
                                    str(channels_in), str(channels_out), 
                                    normalized_weights, weights_dist, renorm] 


                    df = pd.DataFrame([data_tuning],
                                    columns=[ 'session', 'area1','num1', 'area2', 'num2',   
                                    'cortex1', 'cortex2',   
                                    'str_freq1', 'str_freq2',  
                                    'window_size1', 'window_size2', 
                                    'step', #'delay',  
                                    'n_trial', 'data_size', 
                                    'n_chans1', 'n_chans2', 
                                    'only_correct_trials', 'only_align_on_available',
                                    'mse_test', 'mse_train', 'var(y_test)',  
                                    'r2_test','r2_error_bar', 'var-mse','R2_time', 'time',
                                    'n_iterations', 'n_epochs', 'size_of_batches', 
                                    'learning_rate',
                                    'l2', 'l2_regularization_penalty', 
                                    'kpt', 'keep_prob_train',
                                    'bn', 'DECAY',
                                    'patch_dim', 'pool_dim', 'nonlin',  
                                    'fc_conv', 'fc_central', 'fc_deconv', 
                                    'n_layers', 
                                    'channels_in', 'channels_out', 
                                    'normalized_weights', 'weights_dist', 'renorm'] ,
                                    index=[0])
                                    
                    file_name = '/home/rudy/Python2/predictogram/tuning/' + 'tuning_channel_to_channel.csv'
                    file_exists = os.path.isfile(file_name)
                    if file_exists :
                        with open(file_name, 'a') as f:
                            df.to_csv(f, mode ='a', index=False, header=False)
                    else:
                        with open(file_name, 'w') as f:
                            df.to_csv(f, mode ='w', index=False, header=True)
                                    
                
                
                    
                
                
                
                
                

    