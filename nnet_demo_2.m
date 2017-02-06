function nnet_demo_2

seed = 1234;

randn('state', seed );
rand('twister', seed+1 );


%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.
maxepoch = 500;


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

numchunks = 8;
numchunks_test = 8;
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

numchunks = 20;
numchunks_test = 8;
%%%%%%%%
%}


%{
%%%%%%%%
% CURVES
%%%%%%%%

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

numchunks = 4
numchunks_test = 4;
%%%%%%%%
%}

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;




runDesc = ['seed = ' num2str(seed) ', enter anything else you want to remember here' ];

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set

resumeFile = [];

paramsp = [];
Win = [];
bin = [];
%[Win, bin] = loadPretrainedNet_curves;



mattype = 'gn'; %Gauss-Newton.  The other choices probably won't work for whatever you're doing
%mattype = 'hess';
%mattype = 'empfish';

rms = 0;

hybridmode = 1;

%decay = 1.0;
decay = 0.95;

%jacket = 0;
%this enables Jacket mode for the GPU
jacket = 1;
%gpuDevice(1);

errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

runName = 'test_geoorig';

% nnet_train_ng( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
nnet_train_geo( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);