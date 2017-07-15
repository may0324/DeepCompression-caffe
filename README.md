# DeepCompression-caffe
Caffe for Deep Compression

# Introduction
This is a simple caffe implementation of Deep Compression(https://arxiv.org/abs/1510.00149), including weight prunning and quantization.<br>
According to the paper, the compression are implemented only on convolution and fully-connected layers.<br>
Thus we add a CmpConvolution and a CmpInnerProduct layer.<br>
The params that controlls the sparsity including:<br>
* sparse_ratio: the ratio of pruned weights<br>
* class_num: the numbers of k-means for weight quantization<br>
* quantization_term: whether to set quantization on <br>

For a better understanding, please see the examples/mnist and run the demo script, which we automatically compress a pretrained MNIST　LeNet caffemodel.

# Run LeNet Compressing Demo

```
$Bash
```

```Bash
# clone repository and make 
$ git clone https://github.com/may0324/DeepCompression-caffe.git
$ cd DeepCompression-caffe
$ make -j 32 

# run demo script, this will finetune a pretrained model
$ python examples/mnist/train_compress_lenet.py

```




