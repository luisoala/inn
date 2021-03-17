# gmprop

## Gaussian Moment Propagation

Implementations of several types of neural network layers that propagate
*probability distributions* in terms of first and second moments instead of 
*point-estimates*. More precisely, the moments describe Gaussian densities. 
This method is also called *assumed density filtering* (ADF) with 
Gaussian densities.

The method is described in more detail in 
"Lightweight Probabilistic Deep Networks, Jochen Gast & Stefan Roth, 2018".

The implementation of moment proagating neural network layers is based on 
modifications of the corresponding Tensorflow Keras Layers.

The following layers types are curently available:

    * Dense layers
    * Conv2D layers
    * AveragePooling2D layers
    * Reshape, Flatten, Permute layers
    
The following activations functions are currently available:

    * linear identity activation (i.e. no activation)
    * rectified linear unit (ReLU) activation
    
Probabilistic output units require non-standard loss functions that take into 
account the second moments. Several negative log-likelihood based losses can be
used, which differ in the type of probability distribution assumed for the 
final output layer (hidden layers are always considered Gaussian).

Loss functions based on the following output distributions 
are currently available:

    * Gaussian (for regression tasks)
    * Power-Exponential (for regression tasks)
    * Dirichlet (for classification tasks)

## Content of this directory

    * Python package 'gmprop' containing the layer implementations
    * module for parsing and generating different types of test data
    * scripts for training of a standard and the corresponding gmprop network
      for inpainting on generated toy image data (filling in stripes)

