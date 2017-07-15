#!/usr/bin/env python

import sys
import os
"""
This is a script to automatically compressing a pretrained mnist lenet caffemodel, including weight prunnng and weight quantization.
In practice, the layers are much more sensitive to weight prunning than weight quantization. So we suggest to do weight prunning layer-wisely 
and do weight quantization finally since it almost does no harm to accuary.
In this script, we set the sparse ratio (the ratio of pruned weights) layer-wisely and do each finetuning iteration.
After all layers are properly pruned, weight quantization are done on all layers simultaneously.
The final accuracy of finetuned model is about 99.06%, you can check if the weights are most pruned and weight-shared for sure.
Enjoy!

Please refer to http://blog.csdn.net/may0324/article/details/52935869 for more. 
"""



sparse_ratio_vec = [0.33, 0.8, 0.9, 0.8] #sparse ratio of each layer
iters = [500, 1000, 10500, 11000, 500] #max iteration of each stage

def generate_data_layer():
    data_layer_str = '''
name: "LeNet"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100 
    backend: LMDB
  }
}

    '''

    return data_layer_str

def generate_cmp_conv_layer(kernel_size, kernel_num, stride,  layer_name, bottom, top, filler="xavier", sparse_ratio=0, class_num=256, quantize_term="false"):
    tmp =''
    if filler == 'gaussian':
      tmp = '''    std: 0.01
	  '''
    conv_layer_str = '''
layer {
  name: "%s"
  type: "CmpConvolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
  }
  param {
     lr_mult: 2
  }
  convolution_param {
    num_output: %d
    kernel_size: %d
    stride: %d
    sparse_ratio: %f
    class_num: %d
    quantize_term: %s
    weight_filler {
      type: "%s"
    ''' %(layer_name, bottom, top, kernel_num, kernel_size, stride, sparse_ratio, class_num, quantize_term, filler) + tmp + '''
    }
    bias_filler {
      type: "constant"
    }
  }
}
    '''
    return conv_layer_str
def generate_cmp_fc_layer(kernel_num, layer_name, bottom, top, filler="xavier", sparse_ratio=0, class_num=256, quantize_term="false"):
    tmp =''
    if filler == 'gaussian':
      tmp = '''    std: 0.01
	  '''
    fc_layer_str = '''
layer {
  name: "%s"
  type: "CmpInnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: %d
    sparse_ratio: %f
    class_num: %d
    quantize_term: %s
    weight_filler {
      type: "%s"
    ''' %(layer_name, bottom, top, kernel_num, sparse_ratio, class_num, quantize_term, filler ) + tmp + '''
    }
    bias_filler {
      type: "constant"
    }
  }
}'''
    return fc_layer_str
def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''
layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str


def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''
layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy"
  include {
     phase: TEST
  }
}

'''%(bottom, bottom)
    return softmax_loss_str


def generate_lenet(stage):
    if stage<1:
       return ''
    network_str = generate_data_layer()
    if stage == 5: #last stage do weight quantization
      quantize_term = "true"
    else:
      quantize_term = "false"
    
    ratio = sparse_ratio_vec[0]
    network_str += generate_cmp_conv_layer(5,20,1,"conv1","data","conv1","xavier",ratio,256,quantize_term)
    network_str += generate_pooling_layer(2,2,"MAX","pool1","conv1","pool1")
    if stage >= 2:
      ratio = sparse_ratio_vec[1]
    else:
      ratio = 0
    network_str += generate_cmp_conv_layer(5,50,1,"conv2","pool1","conv2","xavier",ratio,256,quantize_term)
    network_str += generate_pooling_layer(2,2,"MAX","pool2","conv2","pool2")
    if stage >= 3:
      ratio = sparse_ratio_vec[2]
    else:
      ratio = 0
    network_str += generate_cmp_fc_layer(500,"fc1","pool2","fc1","xavier",ratio,32,quantize_term)
    network_str += generate_activation_layer("relu1", "fc1", "fc1", "ReLU")
    if stage >= 4:
      ratio = sparse_ratio_vec[3]
    else:
      ratio = 0
    network_str += generate_cmp_fc_layer(10,"fc2","fc1","fc2","xavier",ratio,32,quantize_term)
    network_str += generate_softmax_loss("fc2")

    protoname = 'examples/mnist/lenet_train_test_compress_stage%d.prototxt' %stage
    fp = open(protoname, 'w')
    fp.write(network_str)
    fp.close()

def generate_solver(stage, max_iter):
    solver_str = '''
# The train/test net protocol buffer definition
net: "examples/mnist/lenet_train_test_compress_stage%d.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100 
# Carry out testing every 500 training iterations.
test_interval: 500 
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.001#0.01
momentum: 0.9 
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100 
# The maximum number of iterations
max_iter: %d 
# snapshot intermediate results
snapshot: 500 
snapshot_prefix: "examples/mnist/models/lenet_finetune_stage%d" 
# solver mode: CPU or GPU
solver_mode: GPU 

''' %(stage, max_iter, stage ) 
    protoname = 'examples/mnist/lenet_solver_stage%d.prototxt' %stage
    fp = open(protoname,'w')
    fp.write(solver_str)
    fp.close()
   
if __name__ == '__main__':
    
    max_stage = 5 
    for s in range(0,max_stage):
      generate_lenet(s+1)
      generate_solver(s+1,iters[s])
      
      if s==0:
        modelfile = "lenet_iter_10000.caffemodel" #initial model
      else:
        modelfile = "lenet_finetune_stage%d_iter_%d.caffemodel" %(s, iters[s-1]) #model of last stage
      cmd = "./build/tools/caffe train --solver=examples/mnist/lenet_solver_stage%d.prototxt --weights=examples/mnist/models/%s " %(s+1, modelfile)
      #print cmd
      os.system(cmd)      


