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

For a better understanding, please see the examples/mnist and run the demo script, which automatically compresses a pretrained MNIST LeNet caffemodel.

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

# Details 
the sparse parameters of lenet are set based on the paper as follows:<br>
---------------------------------------------  <br>
|layer_name | sparse_ratio | quantization_num| <br>
---------------------------------------------  <br>
|   conv1   |     0.33     |       256       | <br>
---------------------------------------------  <br>
|   conv2   |     0.8      |       256       | <br>
---------------------------------------------  <br>
|    fc1    |     0.9      |       32        | <br>
---------------------------------------------  <br>
|    fc2    |     0.8      |       32        | <br>
---------------------------------------------  <br>

In practice, the layers are much more sensitive to weight prunning than weight quantization. <br>
So we suggest to do weight prunning layer-wisely 
and do weight quantization finally since it almost does no harm to accuary. <br>
In the script demo, we set the sparse ratio (the ratio of pruned weights) layer-wisely and do each finetuning iteration.
After all layers are properly pruned, weight quantization are done on all layers simultaneously. <br>
The final accuracy of finetuned model is about 99.06%, you can check if the weights are most pruned and weight-shared for sure.<br>

Enjoy! 

