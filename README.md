# What: Geodesic Acceleration being implemented with base of Martens et al. code.

# Why: Faster optimization on difficult deep networks

# Used parts of (or inspired by) the following papers/repos:

http://www.cs.toronto.edu/~jmartens/docs/HF_book_chapter.pdf (associated with HFdemo.zip)

https://arxiv.org/pdf/1010.1449v1.pdf (Geodesic acceleration applied to non-linear least squares)

https://arxiv.org/pdf/1503.05671.pdf (updated KFAC method by Martens)

http://www.cs.toronto.edu/~jmartens/research.html (Codes HFDemo.zip and KFAC3-MATLAB.zip)


# Requirements

* Matlab

# To run:

* Download data: wget www.cs.toronto.edu/~jmartens/digs3pts_1.mat

* Open in Matlab: nnet_demo_2.m file

* Hit Run

# Expected results

* Geodesic acceleration helps reach smaller training error faster than natural gradient on deep networks

# Issues

* May have to set jacket to 0 or compMode to 'cpu_single' to work.
