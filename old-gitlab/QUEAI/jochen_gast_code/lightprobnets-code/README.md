# Lightweight Probabilistic Deep Networks

This source code accompanies the paper:<br>

**[Lightweight Probabilistic Deep Networks](https://arxiv.org/abs/1805.11327)**, [Jochen Gast](http://www.visinf.tu-darmstadt.de/team_members/jgast/jgast.en.jsp) and [Stefan Roth](http://www.visinf.tu-darmstadt.de/team_members/sroth/sroth.en.jsp), CVPR 2018.

## Framework Requirements
This code has been built and tested with PyTorch 0.4.0 on Python 2.7. Please check [PyTorch](https://pytorch.org) for requirements and setup instructions. Typically, the easiest way is to install a full [Anaconda Python distribution](https://www.anaconda.com/download/).

## Custom packages
After having installed Anaconda, following custom packages are required:
``setproctitle tqdm``.
They can be installed via the one-liner

```
$ pip install setproctitle tqdm
```

## Compile LogGammaModule
For Dirichlet layers, a custom LogGammaModule has to be compiled via the nvcc compiler. An appropriate Makefile can be executed at:

```
/lightprobnets/contrib/lgamma_function$ make
```

## Scripts
In /scripts you find various training/testing scripts that can be run via

```
lightprobnets/scripts$ . train_mnist_lenet_probout.sh
lightprobnets/scripts$ . train_mnist_lenet_adf.sh
lightprobnets/scripts$ . train_flyingchairs_flownetprobout.sh
...
```
Within each script you must specify dataset locations via environment variables. Small datasets such as CIFAR or MNIST are downloaded automatically if not existing. FlyingChairs and Sintel have to be downloaded beforehand.


## Contact
If you have further questions, feel free to contact [jochen.gast@visinf.tu-darmstadt.de](mailto:jochen.gast@visinf.tu-darmstadt.de) .

## Citation
If you use our code, please cite our CVPR 2018 paper:

```
@inproceedings{Gast:2018:LPD,
  title={Lightweight Probabilistic Deep Networks},
  author={Jochen Gast and Stefan Roth},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```
