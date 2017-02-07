function KFAC_demo_2

seed = 1234;

randn('state', seed );
rand('twister', seed+1 );


maxiter = 10000;

%uncomment the appropriate section to use a particular dataset


%%%%%%%%
% MNIST
%%%%%%%%

%dataset available at www.cs.toronto.edu/~jmartens/mnist_all.mat

load mnist_all
traindata = zeros(0, 28^2);
for i = 0:9
    eval(['traindata = [traindata; train' num2str(i) '];']);
end
%indata = double(traindata)/255;
indata = single(traindata)/255;
clear traindata

testdata = zeros(0, 28^2);
for i = 0:9
    eval(['testdata = [testdata; test' num2str(i) '];']);
end
%intest = double(testdata)/255;
intest = single(testdata)/255;
clear testdata

indata = indata';
intest = intest';

perm = randperm(size(intest,2));
intest = intest( :, perm );

randn('state', seed );
rand('twister', seed+1 );

perm = randperm(size(indata,2));
indata = indata( :, perm );

layersizes = [1000 500 250 30 250 500 1000];
layertypes = {'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic'};

%standard L_2 weight-decay:
weightcost = 1e-5;

runName = 'KFAC3_mnist_1';
runDesc = '[[version 5]], expected output for MNIST';

%%%%%%%%



%{
%%%%%%%%
% FACES
%%%%%%%%

%dataset available at www.cs.toronto.edu/~jmartens/newfaces_rot_single.mat

load newfaces_rot_single
total = 165600;
trainsize = (total/40)*25;
testsize = (total/40)*10;
indata = newfaces_single(:, 1:trainsize);
intest = newfaces_single(:, (end-testsize+1):end);
clear newfaces_single


perm = randperm(size(intest,2));
intest = intest( :, perm );
%randn('state', seed );
%rand('twister', seed+1 );

perm = randperm(size(indata,2));
%disp('Using 1/2');
%perm = perm( 1:size(indata,1)/2 );
indata = indata( :, perm );
%outdata = outdata( :, perm );

layertypes = {'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'linear'};
layersizes = [2000 1000 500 30 500 1000 2000];

%standard L_2 weight-decay:
weightcost = 1e-5;
weightcost = weightcost / 2; %an older version of the code used in the paper had a differently scaled objective (by 2) in the case of linear output units.  Thus we now need to reduce weightcost by a factor 2 to be consistent
%%%%%%%%

runName = 'KFAC3_faces_1';
runDesc = '[[version 5]], expected output for FACES';
%}



%%%%%%%%
% CURVES
%%%%%%%%
%{
%dataset available at www.cs.toronto.edu/~jmartens/digs3pts_1.mat

tmp = load('digs3pts_1.mat');
indata = tmp.bdata';
%outdata = tmp.bdata;
intest = tmp.bdatatest';
%outtest = tmp.bdatatest;
clear tmp

perm = randperm(size(indata,2));
%disp('Using 1/2');
%perm = perm( 1:size(indata,1)/2 );
indata = indata( :, perm );
%outdata = outdata( :, perm );

layersizes = [400 200 100 50 25 6 25 50 100 200 400];
layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic'};

%standard L_2 weight-decay:
weightcost = 1e-5;
%%%%%%%%

runName = 'KFAC3_curves_1';
runDesc = '[[version 5]], expected output for CURVES';

%}




%for autoencoders target output = input:
outdata = indata;
outtest = intest;






resumeFile = [];

paramsp = [];
Win = [];
bin = [];


%matrix type used in exact quadratic model:
mattype = 'gn'; %should always be this
%mattype = 'hess'

rms = 0;

%storing the activities on the GPU (assuming this is being used) instead of the CPU
cacheOnGPU = 1;
%cacheOnGPU = 0; %this will save memory but possibly slow things down


%how to perform computations
%compMode = 'cpu_single'; %standard cpu computation
%compMode = 'jacket'; %Jacket package (tests on autoencoders from the paper were done using this)
compMode = 'gpu_single'; %built-in MATLAB gpu stuff.  My tests indicate this is *much* slower than Jacket, and gives very low GPU utilization vs Jacket or various Python-based GPU packages.  Unfortunately you can't buy Jacket anymore because of Mathwork's litigious nature.  Please do not use this mode to benchmark K-FAC.
%compMode = 'gpu_double';

%the error function to report (which is different from the objective, which is assumed to by the "matching" log-likelihood function associated with the output units)
errtype = 'L2'; %L2-norm error
%errtype = 'class'; %classification error



%NOTE: all of these parameters to do with mini-batch will be somewhat problem dependent.  The settings below are what we used in autoencoder experiments in the paper

%the maximum number of cases to be processed by the GPU at once:
maxchunksize = 5000; 

%the starting size of the minibatch:
minibatch_startsize = 1000; 

%the final size of the minibatch:
%minibatch_maxsize = 10000;
minibatch_maxsize = size(indata,2);  %ending on full batch will not always be the best strategy.  It was for the autoencoders though

%target iteration for maximum size reached using exponential schedule (set to 1 for a constant size = minibatch_maxsize)
%minibatch_maxsize_targetiter = 1;
%minibatch_maxsize_targetiter = 250;
minibatch_maxsize_targetiter = 500; %there is a lot of room for tuning here.  Really the mini-batch size should probably be adapted intelligently somehow (e.g. as in the Byrd et al. paper)
%minibatch_maxsize_targetiter = 1000;


% KFAC
nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxiter, indata, outdata, intest, outtest, layersizes, layertypes, mattype, rms, errtype, weightcost, minibatch_startsize, minibatch_maxsize, minibatch_maxsize_targetiter, maxchunksize, compMode, cacheOnGPU);

