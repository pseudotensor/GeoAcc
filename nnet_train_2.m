function paramsp = nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxiter, indata, outdata, intest, outtest, layersizes, layertypes, mattype, rms, errtype, weightcost, minibatch_startsize, minibatch_maxsize, minibatch_maxsize_targetiter, maxchunksize, compMode, cacheOnGPU)

%Companion code for the paper "Optimizing Neural Networks with Kronecker-Factored Approximate Curvature".
%
% Note that the notation used in the code is very different from the paper.  The code was derived from the code for the paper "Deep Learning via Hessian-free Optimization" which used different notational conventions.  The following is a rough guide for understanding how the notation in the code is related to that in the paper:
% 
% - activities are denoted by y (a in the paper)
% - unit inputs are denoted by x (s in the paper)
% - y_hom means y with the homogenous coordinate appended
% - 2 after most quantities, as in y_hom2 or dEdx2 denotes a 2nd-order statistic (expected outer product)
% - deritivates with respect to the negative loss of some quantity A of is denoted by dEdA
% - indices after quantities usually refer to layer.  But unlike in the paper, the index goes up between the input to a layer and its output.  i.e. x_{i-1} is the input to layer i, and y_i is the output
%
% Also note that the design used in this implementation is quite primitive.  The bulk of the length of the code is actually various re-implementations of the standard forwards and backwards pass for neural nets that each compute slightly different things.  A more elegant design could get away with implementing the forwards and backwards pass only once.
%
%
% There are two main things to pay attension to when using K-FAC.  
% 
% First, the initial lambda value ("initlambda" below) should be chosen appropriately.  You should only need to run the method for a few dozen iterations to see if you choice is good.  Too low or too high and the method will make progress very slowly, until the automatic adjustment routines have time to compensate.  You can usually tell which direction to adjust lambda based on what these routines do.  As a rule of thumb the initial lambda value will be lower for classificatio nets than for autoencoders.  Note that it should be easy to automate this initial tuning of lambda, I just haven't bothered with it.
% 
% Second, the choice of the mini-batch schedule can be pretty important for optimal performance.  The part of K-FAC that automatically determines the learning rate and momentum decay constant (alpha and mu) doesn't work well in the presence of lots of noise in the gradient.  As optimization proceeds and the signal-to-noise ratio gets low, you should use more data to compensate.  Using too much data early on will also be wasteful.  And it won't always be appropriate to reach full batch mode by some fixed iteration as in the autoencoder experiments.  Note that an alternative to increasing the mini-batch size is to switch to using fixed values of alpha and mu after some point in the optimization, but these would have to be tuned.


disp( ['Starting run named: ' runName ]);

rec_constants = {'layersizes', 'rms', 'weightcost', 'autodamp', 'initlambda', 'lambda_adj_interval', 'drop', 'boost', 'initgamma', 'gamma_adj_interval', 'gamma_drop', 'gamma_boost', 'mattype', 'errtype', 'stochastic_mode', 'numsamples', 'minibatch_startsize', 'minibatch_maxsize', 'minibatch_maxsize_targetiter', 'maxchunksize', 'psize', 'useTriBlock', 'useEIGs', 'ratio_sample', 'ratio_BV', 'refreshInvApprox_interval', 'maxfade_stats', 'maxfade_avg', 'avg_start', 'useMomentum', 'compMode' };


%Fraction of data from mini-batch to use in estimating 2nd-order statistics used in Fisher approximation
ratio_sample = 1/8;
%ratio_sample = 1

%Fraction of data from mini-batch to use in computing matrix-vector products with exact Fisher (when deterministic learning rate and momentum decay constant)
%ratio_BV = 1/4; %using a value below 1 is dangerous and should be done with caution
ratio_BV = 1;


%Use block-tridiagonal approximation (vs block-diagonal)
useTriBlock = 1;
%useTriBlock = 0

%Use the automatically calibrated momentum discussed in the paper
useMomentum = 1;
%useMomentum = 0;

%Use eigenvalue decompositions in favor of inverses (saves some work at the cost of introducing more work.  Trade-off depends on various factors and other metaparameters)
useEIGs = 1;
%useEIGs = 0

%Compute off-tridiagonal blocks of Fisher (only used in debugging)
computeOffTriDiag = 0;
%computeOffTriDiag = 1

%Compute and report averaged parameters.  Averaging is much more useful for SGD, and is useful in K-FAC usually only when the mini-batch size doesn't grow.  You should always look at the best between averaged and non-averaged, since the non-averaged can be (much) better sometimes.
reportAveragedParams = 1;

%Start averaging at this iteration
avg_start = 200;
%Exponentially decay running averaged iterate by this value
maxfade_avg = 0.99;


%Recompute the approximate Fisher after this many iterations (T_3 from the paper)
refreshInvApprox_interval = 20;
%refreshInvApprox_interval = 1
%refreshInvApprox_interval = 5

%Adjust gamma using a local 3-point grid search after this many iterations (T_2 from the paper)
%gamma_adj_interval = Inf
gamma_adj_interval = 20;
%gamma_adj_interval = 5
%gamma_adj_interval = 1

%Adjust lambda using the Levenberg-Marquardt rule after this many iterations (T_1 from the paper)
%lambda_adj_interval = Inf;
lambda_adj_interval = 5;
%lambda_adj_interval = 1


%Compute some diagonal statistics if this is useful to you
computeGrad2 = 0;
%computeGrad2 = 1


%Sample gradients using the model's distribution (as implied by the loss function).  Turning this off uses the empirical Fisher, which is not recommended but might be okay (and can save a bit of work)
stochastic_mode = 1;
%stochastic_mode = 0;


%When computing samples from the model repeat this many times (it is better to increase ratio_sample instead of using this, only ratio_sample is already maxed out [at 1.0])
%numsamples = 10;
%numsamples = 5;
numsamples = 1;


%The rate of exponential decay used for estimating the statistics
maxfade_stats = 0.95;


%The initial lambda value.  Will be problem dependent and should be adjusted based on behavior of first few dozen iterations.  One could conceivably automate this initial adjustment
initlambda = 150
%initlambda = 1  %a value usually more appropriate for classification nets

%Multiplies to increase and decrease lambda when being adjusted.  NOTE: the code automatically raises this to the power lambda_adj_interval to compensate for how occasional adjustment would otherwise slow things down
drop = 19/20;
%drop = 4/5;
boost = 1/drop;

lambda_max = Inf;
lambda_min = 0;
%lambda_min = 1e-3


%Below are similar constants as the case for lambda, but for the related meta-parameter gamma.  Note that gamma is going to be at a scale sqrt that of lambda typically, hence the sqrt's you see below
initgamma = sqrt(initlambda + weightcost);
%initgamma = 500

gamma_drop = sqrt(drop);
gamma_boost = 1/gamma_drop;

gamma_max = 1; %completely arbitrary, but prevents gamma from sometimes getting stuck at a too-high value for a while.  use with caution
gamma_min = sqrt(weightcost);


autodamp = 1;


checkpoint_every_iter = 500;
save_every_iter = 1;
report_full_obj_every_iter = 50;



storeD = 0;

if strcmp(mattype, 'hess')
    computeBV = @computeHV;
    storeD = 1;
elseif strcmp(mattype, 'gn')
    computeBV = @computeGV;
    computeBfactV = @computeGfactV;
%elseif strcmp(mattype, 'empfish')
%    computeBV = @computeFV;
end




function [V,D] = geig(mat)
    [V,D] = eig(single(mat));
    V = conv(V);
    D = conv(D);
end


function [U,D] = gsvd(mat)
    [U,D] = svd(single(mat));
    U = conv(U);
    D = conv(D);
end



if strcmp( compMode, 'jacket' )
    
    conv = @gsingle;

    mones = @gones;
    mzeros = @gzeros;
    meye = @geye;
    
    mrandn = @grandn;
    mrand = @grand;
    
    %do random numbers on the CPU instead (seems slower):
    %mrandn = @(varargin) conv( randn(varargin{:}, 'single') );
    %mrand = @(varargin) conv( rand(varargin{:}, 'single') );
    %disp('rand on CPU!!')
    %pause
    
    minv = @(mat)conv(inv(double(mat)));
    %minv = @(mat)conv(inv(single(mat))); %more efficient?
    meig = @geig;
    mchol = @(mat) conv(chol(double(mat)));
    msvd = @gsvd;

    if cacheOnGPU
        store = conv;
    else
        store = @single;
    end
    
    makeDouble = @double;


elseif strcmp( compMode, 'gpu_single' )

    conv = @gpuArray;

    mones = @(varargin) ones(varargin{:}, 'single', 'gpuArray');
    mzeros = @(varargin) zeros(varargin{:}, 'single', 'gpuArray');
    meye = @(varargin) eye(varargin{:}, 'single', 'gpuArray');

    minv = @inv;
    meig = @eig;
    mchol = @chol;
    msvd = @svd;

    mrand = @(varargin) rand(varargin{:}, 'single', 'gpuArray');
    mrandn = @(varargin) randn(varargin{:}, 'single', 'gpuArray');

    if cacheOnGPU
        store = conv;
    else
        store = @gather;
    end

    makeDouble = @(x) double(gather(x));

elseif strcmp( compMode, 'gpu_double' )

    conv = @gpuArray;

    mones = @(varargin) ones(varargin{:}, 'double', 'gpuArray');
    mzeros = @(varargin) zeros(varargin{:}, 'double', 'gpuArray');
    meye = @(varargin) eye(varargin{:}, 'double', 'gpuArray');
    
    minv = @inv;
    meig = @eig;
    mchol = @chol;
    msvd = @svd;

    mrand = @(varargin) rand(varargin{:}, 'double', 'gpuArray');
    mrandn = @(varargin) randn(varargin{:}, 'double', 'gpuArray');
    
    if cacheOnGPU
        store = conv;
    else
        store = @gather;
    end

    makeDouble = @(x) double(gather(x));
    
    
elseif strcmp( compMode, 'cpu_single' )

    conv = @single;
    
    mones = @(varargin) ones(varargin{:}, 'single');
    mzeros = @(varargin) zeros(varargin{:}, 'single');
    meye = @(varargin) eye(varargin{:}, 'single');

    minv = @inv;
    meig = @eig;
    mchol = @chol;
    msvd = @svd;

    mrand = @rand;
    mrandn = @randn;

    store = conv;

    makeDouble = @double;
    
    
elseif strcmp( compMode, 'cpu_double' )

    conv = @double;
    
    mones = @ones;
    mzeros = @zeros;
    meye = @eye;

    minv = @inv;
    meig = @eig;
    mchol = @chol;
    msvd = @svd;

    mrand = @rand;
    mrandn = @randn;

    store = conv;
    
    makeDouble = @double;
    
    
else

    error('invalid compMode value');

end




%softplus = @(x)(log(1+exp(x.*(x<=50))).*(x<=50) + x.*(x>50));
%softplus = @(x)((x>=0).*x + log(1+exp((2.*(x<0)-1).*x))); 
softplus = @(x)((x>=0).*x + log(1+exp(-abs(x))));
dsoftplus = @(y)(1 - 1./exp(y));

function samp = samplesoftmax(p)

    c = cumsum(p,1);

    s = mrand(1,size(p,2));

    samp = diff( [mzeros(1,size(p,2)) ; repmat(s, [size(p,1) 1]) <= c] );

end


layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims numcases] = size(indata);
[tmp numtest] = size(intest);


function v = vec(A)
    v = A(:);
end


psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

%pack all the parameters into a single vector for easy manipulation
function M = pack(W,b)
    
    M = mzeros( psize, 1 );
    
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
        
        %cur
    end
    
end

%unpack parameters from a vector so they can be used in various neural-net
%computations
function [W,b] = unpack(M)

    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
    
    cur = 0;
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );

        cur = cur + layersizes(i)*layersizes(i+1);
        
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );

        cur = cur + layersizes(i+1);
    end
    
end



%Hessian-vector product (only used for testing purposes)
function HV = computeHV(V)
    
    [VWu, Vbu] = unpack(V);
    
    HV = mzeros(psize,1);

    
    for chunk = 1:numchunks_BV

        %application of R operator
        Ry = cell(numlayers+1,1);
        RdEdy = cell(numlayers+1,1);
        RdEdx = cell(numlayers, 1);
        HVW = cell(numlayers,1);
        HVb = cell(numlayers,1);

        yip1 = conv(y{1,chunk}(:,1:sizechunk_BV(chunk)));

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk_BV(chunk));        
        
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk_BV(chunk)]);

            Ry{i} = store(Ryi);
            Ryi = [];

            yip1 = conv(y{i+1,chunk}(:,1:sizechunk_BV(chunk)));
            
            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
            else
                error( 'Unknown/unsupported layer type' );
            end

            Rxi = [];

        end


        %backward prop:
        %cross-entropy for logistics:
        %RdEdy{numlayers+1} = (-(outdata./(y{numlayers+1}.^2) + (1-outdata)./(1-y{numlayers+1}).^2)).*Ry{numlayers+1};
        %cross-entropy for softmax:
        %RdEdy{numlayers+1} = -outdata./(y{numlayers+1}.^2).*Ry{numlayers+1};
        for i = numlayers:-1:1

            if i < numlayers
                if strcmp(layertypes{i}, 'logistic')
                    %logistics:

                    dEdyip1 = conv( dEdy_emp{i+1,chunk}(:,1:sizechunk_BV(chunk)) );
                    RdEdx{i} = RdEdy{i+1}.*yip1.*(1-yip1) + dEdyip1.*Ryip1.*(1-2*yip1);
                    dEdyip1 = [];

                elseif strcmp(layertypes{i}, 'linear')
                    RdEdx{i} = RdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end

            else
                if ~rms
                    %assume canonical link functions:
                    RdEdx{i} = -Ryip1;
                else
                    error('disabled');
                end
            end
            RdEdy{i+1} = [];

            yip1 = []; Ryip1 = [];

            yi = conv(y{i,chunk}(:,1:sizechunk_BV(chunk)));
            Ryi = conv( Ry{i} );        

            dEdxi = conv( dEdx_emp{i,chunk}(:,1:sizechunk_BV(chunk)) );

            RdEdy{i} = VWu{i}'*dEdxi + Wu{i}'*RdEdx{i};

            %(HV = RdEdW)
            HVW{i} = RdEdx{i}*yi' + dEdxi*Ryi';
            HVb{i} = sum(RdEdx{i},2);

            RdEdx{i} = []; dEdxi = [];

            yip1 = yi; yi = [];
            Ryip1 = Ryi; Ryi = [];

        end
        yip1 = []; Ryip1 = []; RdEdy{1} = [];


        HV = HV + pack(HVW, HVb);
    end

    HV = HV / conv(numcases);
    
    HV = HV - conv(weightcost)*(maskp.*V);
    
    if autodamp
        %HV = HV - conv(lambda)*V;
    end
    
end




%Fisher/Gauss-Newton-vector product (only used for testing correctness of factored product function)
function GV = computeGV(V)

    [VWu, Vbu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    for chunk = 1:numchunks_BV

        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        yip1 = conv(y{1,chunk}(:,1:sizechunk_BV(chunk)));

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk_BV(chunk));

        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk_BV(chunk)]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{i+1,chunk}(:,1:sizechunk_BV(chunk)));

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'tanh')
                Ryip1 = Rxi.*(1+yip1).*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
            elseif strcmp( layertypes{i}, 'softplus' )
                Ryip1 = Rxi.*dsoftplus(yip1);
            else
                error( 'Unknown/unsupported layer type' );
            end

            Rxi = [];

        end

        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                elseif strcmp( layertypes{i}, 'softplus' )
                    rdEdx{i} = rdEdy{i+1}.*dsoftplus(yip1);
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    %assume canonical link functions:
                    rdEdx{i} = -Ryip1;
                else
                    RdEdyip1 = -2*Ryip1;

                    if strcmp(layertypes{i}, 'softmax')
                        error( 'RMS error not supported with softmax output' );
                    elseif strcmp(layertypes{i}, 'logistic')
                        rdEdx{i} = RdEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        rdEdx{i} = RdEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        rdEdx{i} = RdEdyip1;
                    elseif strcmp( layertypes{i}, 'softplus' )
                        rdEdx{i} = RdEdyip1.*dsoftplus(yip1);
                    else
                        error( 'Unknown/unsupported layer type' );
                    end

                    RdEdyip1 = [];

                end

                Ryip1 = [];

            end
            rdEdy{i+1} = [];

            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{i,chunk}(:,1:sizechunk_BV(chunk)));

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb);

    end
    
    GV = GV / conv(size_BVminibatch);

    GV = GV - conv(weightcost)*(maskp.*V);

    if autodamp
        GV = GV - conv(lambda)*V;
        %GV = GV - conv(lambda^2)*V;
        %GV = GV - conv(lambda)*compute_triBlkV(V);
    end
    
end



%factorized Fisher-vector product
function GfactV = computeGfactV(V,chunk)

    [VWu, Vbu] = unpack(V);

    if chunk > 0 && chunk <= numchunks_BV
        
        yip1 = conv(y{1,chunk}(:,1:sizechunk_BV(chunk)));

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk_BV(chunk));

        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk_BV(chunk)]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{i+1,chunk}(:,1:sizechunk_BV(chunk)));

            if i < numlayers || rms
                if strcmp(layertypes{i}, 'logistic')
                    Ryip1 = Rxi.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    Ryip1 = Rxi.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    Ryip1 = Rxi;
                elseif strcmp( layertypes{i}, 'softmax' )
                    Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
                elseif strcmp( layertypes{i}, 'softplus' )
                    Ryip1 = Rxi.*dsoftplus(yip1);
                else
                    error( 'Unknown/unsupported layer type' );
                end
                
                if rms
                    Ryip1 = sqrt(2)*Ryip1; %this part should probably be checked
                end
                
            else
                if strcmp(layertypes{i}, 'logistic')
                    Ryip1 = Rxi.*sqrt(yip1.*(1-yip1));
                elseif strcmp(layertypes{i}, 'tanh')
                    Ryip1 = Rxi.*sqrt((1+yip1).*(1-yip1));
                elseif strcmp(layertypes{i}, 'linear')
                    Ryip1 = Rxi;
                elseif strcmp( layertypes{i}, 'softmax' )
                    %this uses (I - uu^T)^(1/2) = I - uu^T for ||u||_2 = 1
                    Ryip1 = Rxi.*sqrt(yip1) - sqrt(yip1).*repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
                elseif strcmp( layertypes{i}, 'softplus' )
                    Ryip1 = Rxi.*sqrt(dsoftplus(yip1));
                else
                    error( 'Unknown/unsupported layer type' );
                end
            end

            Rxi = [];

        end
        yip1 = [];

        GfactV = vec(Ryip1) / conv(sqrt(size_BVminibatch));

    elseif chunk == 0
        
        if autodamp
            GfactV = sqrt(conv(weightcost)*maskp + conv(lambda)).*V;
        else
            GfactV = sqrt(conv(weightcost)*maskp).*V;
        end
        
    else
        error('bad argument: chunk');
    end
    
end





%Block-diagonal approx Fisher-vector product (only used for testing purposes)
function BlkV = compute_BlkV(V)
    
    [VW, Vb] = unpack(V);

    BlkVW = cell(numlayers, 1);
    BlkVb = cell(numlayers, 1);

    %[gW, gB] = unpack(puregrad);
    
    for i = 1:numlayers

        tmp = dEdx2_damp{i,i}*[VW{i} Vb{i}]*y_hom2_damp{i,i};

        BlkVW{i} = tmp(:,1:end-1);
        BlkVb{i} = tmp(:,end);

        tmp = [];
    end

    BlkV = pack( BlkVW, BlkVb );
    
end


%Block-diagonal approx inverse Fisher-vector product
function invBlkV = compute_invBlkV(V)
    
    [VW, Vb] = unpack(V);

    invBlkVW = cell(numlayers, 1);
    invBlkVb = cell(numlayers, 1);

    %[gW, gb] = unpack(puregrad);
    %[g2W, g2b] = unpack(grad2);
    
    for i = 1:numlayers

        tmp = inv_dEdx2_damp{i}*[VW{i} Vb{i}]*inv_y_hom2_damp{i};
        
        %"proper" Tikhonov approach.  Can be implemented efficiently in the block diagonal case as discussed in the paper
        %tmp = U_dEdx2{i}*((U_dEdx2{i}'*[VW{i} Vb{i}]*U_y_hom2{i})./(D_dEdx2{i}*D_y_hom2{i}' + conv(gamma^2)))*U_y_hom2{i}';
        
        invBlkVW{i} = tmp(:,1:end-1);
        invBlkVb{i} = tmp(:,end);

        tmp = [];
    end

    invBlkV = pack( invBlkVW, invBlkVb );
    
end



%Block-tridiagonal approx inverse Fisher-vector product
function invTriBlkV = compute_invTriBlkV(V)
    
    [VW, Vb] = unpack(V);
    

    V1{numlayers} = [VW{numlayers} Vb{numlayers}];
    for i = 1:numlayers-1
        V1{i} = [VW{i} Vb{i}] - predweights_dEdx{i+1}*[VW{i+1} Vb{i+1}]*predweights_y_hom{i+1}';
    end
    
    %invcondCov{i,i} = inv( Cblk{i,i} - regress{i-1,i}'*Cblk{i-1,i-1}*regress{i-1,i} )
    
    %inv( Cblk{i,i} - Cblk{i-1,i}'*inv(Cblk{i-1,i-1})*Cblk{i-1,i} )
    %= inv( chol(Cblk{i,i})'*chol(Cblk{i,i}) - Cblk{i-1,i}'*inv(Cblk{i-1,i-1})*Cblk{i-1,i} )
    %= inv(chol(Cblk{i,i}))*inv( I - inv(chol(Cblk{i,i})')*Cblk{i-1,i}'*inv(Cblk{i-1,i-1})*Cblk{i-1,i}*inv(chol(Cblk{i,i})) )*inv(chol(Cblk{i,i})')
    
    V2{numlayers} = inv_dEdx2_damp{numlayers}'*V1{numlayers}*inv_y_hom2_damp{numlayers};

    for i = 1:numlayers-1
        tmp = 1 - DM_dEdx{i}*DM_y_hom{i}';
        tmp(tmp == 0) = 1; %this is a hack
        V2{i} = K_dEdx{i}*((K_dEdx{i}'*V1{i}*K_y_hom{i})./tmp)*K_y_hom{i}';
    end
    
    
    V3{1} = V2{1};
    for i = 2:numlayers
        V3{i} = V2{i} - predweights_dEdx{i}'*V2{i-1}*predweights_y_hom{i};
    end
    
    for i = 1:numlayers
        invTriBlkVW{i} = V3{i}(:,1:end-1);
        invTriBlkVb{i} = V3{i}(:,end);
    end
    
    invTriBlkV = pack( invTriBlkVW, invTriBlkVb );
    
end




%inverse Block-Cholesky of block-tridiagonal approx Fisher-vector product (only used for testing)
%it should be the case that compute_invBlkCholTriBlkV_transpose(compute_invBlkCholTriBlkV(V)) = compute_invTriBlkV(V)
function invBlkCholTriBlkV = compute_invBlkCholTriBlkV(V)

    [VW, Vb] = unpack(V);
    
    %output to input version:
    %%%%
    
    V1{numlayers} = [VW{numlayers} Vb{numlayers}];
    for i = 1:numlayers-1
        V1{i} = [VW{i} Vb{i}] - predweights_dEdx{i+1}*[VW{i+1} Vb{i+1}]*predweights_y_hom{i+1}';
    end
    
    
    V2{numlayers} = inv_half_dEdx2_damp{numlayers}'*V1{numlayers}*inv_half_y_hom2_damp{numlayers};

    for i = 1:numlayers-1
        V2{i} = (UM_dEdx{i}*((UM_dEdx{i}'*(  inv_half_dEdx2_damp{i}'*V1{i}*inv_half_y_hom2_damp{i} )*UM_y_hom{i})./(  sqrt(1 - DM_dEdx{i}*DM_y_hom{i}')  ))*UM_y_hom{i}');
    end

    
    for i = 1:numlayers
        invBlkCholTriBlkVW{i} = V2{i}(:,1:end-1);
        invBlkCholTriBlkVb{i} = V2{i}(:,end);
    end
    
    invBlkCholTriBlkV = pack( invBlkCholTriBlkVW, invBlkCholTriBlkVb );
    
end

%transpose of above
function invBlkCholTriBlkV_transpose = compute_invBlkCholTriBlkV_transpose(V)
    
    [VW, Vb] = unpack(V);
    
    %output to input version:
    %%%%
    
    V2{numlayers} = inv_half_dEdx2_damp{numlayers}*[VW{numlayers} Vb{numlayers}]*inv_half_y_hom2_damp{numlayers}';

    for i = 1:numlayers-1
        V2{i} = inv_half_dEdx2_damp{i}*  (UM_dEdx{i}*((UM_dEdx{i}'*( [VW{i} Vb{i}] )*UM_y_hom{i})./(  sqrt(1 - DM_dEdx{i}*DM_y_hom{i}')  ))*UM_y_hom{i}')  * inv_half_y_hom2_damp{i}';
    end
    
    
    V3{1} = V2{1};
    for i = 2:numlayers
        V3{i} = V2{i} - predweights_dEdx{i}'*V2{i-1}*predweights_y_hom{i};
    end
    
    for i = 1:numlayers
        invBlkCholTriBlkVW_transpose{i} = V3{i}(:,1:end-1);
        invBlkCholTriBlkVb_transpose{i} = V3{i}(:,end);
    end
    
    invBlkCholTriBlkV_transpose = pack( invBlkCholTriBlkVW_transpose, invBlkCholTriBlkVb_transpose );
    
end



%Block-tridiagonal approx Fisher-vector product (used for testing compute_invTriBlkV)
function triBlkV = compute_triBlkV(V)
    
    [VW, Vb] = unpack(V);
    
    V1{1} = [VW{1} Vb{1}];
    for i = 2:numlayers
        V1{i} = [VW{i} Vb{i}] + predweights_dEdx{i}'*V1{i-1}*predweights_y_hom{i};
    end

    V2{numlayers} = dEdx2_damp{numlayers,numlayers}'*V1{numlayers}*y_hom2_damp{numlayers,numlayers};

    for i = 1:numlayers-1
        V2{i} = dEdx2_damp{i,i}'*V1{i}*y_hom2_damp{i,i} -  Z_dEdx{i}*V1{i}*Z_y_hom{i};
    end

    
    V3{numlayers} = V2{numlayers};
    for i = numlayers-1:-1:1
        V3{i} = V2{i} + predweights_dEdx{i+1}*V3{i+1}*predweights_y_hom{i+1}';
    end
    
    
    for i = 1:numlayers
        triBlkVW{i} = V3{i}(:,1:end-1);
        triBlkVb{i} = V3{i}(:,end);
    end
    
    triBlkV = pack( triBlkVW, triBlkVb );
    
end

%Block-Cholesky of block-tridiagonal approx Fisher-vector product (only used for testing)
%it should be the case that compute_blkCholTriBlkV(compute_blkCholTriBlkV_transpose(V)) = compute_triBlkV(V)   -  this is the opposite of situation with invBlkChol[...]
function blkCholTriBlkV = compute_blkCholTriBlkV(V)
    
    [VW, Vb] = unpack(V);
    
    
    V2{numlayers} = half_dEdx2{numlayers}'*[VW{numlayers} Vb{numlayers}]*half_y_hom2{numlayers};

    for i = 1:numlayers-1
        V2{i} = half_dEdx2{i}'*  (UM_dEdx{i}*((UM_dEdx{i}'*( [VW{i} Vb{i}] )*UM_y_hom{i}).*sqrt(1 - DM_dEdx{i}*DM_y_hom{i}'))*UM_y_hom{i}')  * half_y_hom2{i};
    end

    
    V3{numlayers} = V2{numlayers};
    for i = numlayers-1:-1:1
        V3{i} = V2{i} + predweights_dEdx{i+1}*V3{i+1}*predweights_y_hom{i+1}';
    end
    
    
    for i = 1:numlayers
        blkCholTriBlkVW{i} = V3{i}(:,1:end-1);
        blkCholTriBlkVb{i} = V3{i}(:,end);
    end
    
    blkCholTriBlkV = pack( blkCholTriBlkVW, blkCholTriBlkVb );
    
end


%transpose of above
function blkCholTriBlkV_transpose = compute_blkCholTriBlkV_transpose(V)
    
    [VW, Vb] = unpack(V);
    
    
    %output to input version:
    %%%%
    
    V1{1} = [VW{1} Vb{1}];
    for i = 2:numlayers
        V1{i} = [VW{i} Vb{i}] + predweights_dEdx{i}'*V1{i-1}*predweights_y_hom{i};
    end

    
    V2{numlayers} = half_dEdx2{numlayers}*V1{numlayers}*half_y_hom2{numlayers}';

    for i = 1:numlayers-1
        V2{i} = UM_dEdx{i}*((UM_dEdx{i}'*(  half_dEdx2{i}*V1{i}*half_y_hom2{i}' )*UM_y_hom{i}).*sqrt(1 - DM_dEdx{i}*DM_y_hom{i}'))*UM_y_hom{i}';
    end

    for i = 1:numlayers
        blkCholTriBlkVW_transpose{i} = V2{i}(:,1:end-1);
        blkCholTriBlkVb_transpose{i} = V2{i}(:,end);
    end
    
    blkCholTriBlkV_transpose = pack( blkCholTriBlkVW_transpose, blkCholTriBlkVb_transpose );
    
end



%more testing code for the 2 layer case
function invTriBlkV = test_compute_invTriBlkV(V)
    
    Fapprox = [ kron( y_hom2_damp{1,1}, dEdx2_damp{1,1} )  kron( y_hom2{1,2}, dEdx2{1,2} ) ;  kron( y_hom2{1,2}', dEdx2{1,2}' )  kron( y_hom2_damp{2,2}, dEdx2_damp{2,2} ) ];
    invTriBlkV = inv(Fapprox)*V;

end


function triBlkV = test_compute_triBlkV(V)
    
    Fapprox = [ kron( y_hom2_damp{1,1}, dEdx2_damp{1,1} )  kron( y_hom2{1,2}, dEdx2{1,2} ) ;  kron( y_hom2{1,2}', dEdx2{1,2}' )  kron( y_hom2_damp{2,2}, dEdx2_damp{2,2} ) ];
    triBlkV = Fapprox*V;
    
end


function fullBlkV = compute_fullBlkV(V)

    if ~computeOffTriDiag
        error('requires computeOffTriDiag = 1');
    end
    
    
    [VW, Vb] = unpack(V);

    for i = 1:numlayers
        
        tmp = mzeros( layersizes(i+1), layersizes(i)+1 );
        
        for j = 1:numlayers
            
            %{
            if i <= j
                tmp = tmp + dEdx2{i,j}*[VW{j} Vb{j}]*y_hom2{i,j}';
            else
                tmp = tmp + dEdx2{j,i}'*[VW{j} Vb{j}]*y_hom2{j,i};
            end
            %}
            if i < j
                tmp = tmp + dEdx2{i,j}*[VW{j} Vb{j}]*y_hom2{i,j}';
            elseif i > j
                tmp = tmp + dEdx2{j,i}'*[VW{j} Vb{j}]*y_hom2{j,i};
            else
                tmp = tmp + dEdx2_damp{i,j}*[VW{j} Vb{j}]*y_hom2_damp{i,j}';
            end
            
        end
        
        fullBlkVW{i} = tmp(:,1:end-1);
        fullBlkVb{i} = tmp(:,end);
        
    end
        
    fullBlkV = pack( fullBlkVW, fullBlkVb );
    
end




function [ll_local, err_local] = computeLL(paramsp_local, indata_local, outdata_local)

    ll_local = 0;
    
    err_local = 0;
    
    [Wu_local,bu_local] = unpack(paramsp_local);
    
    numchunks_local = ceil(size(indata_local,2) / maxchunksize);
    
    %disp( ['numchunks_local = ' num2str(numchunks_local)] );
    
    for chunk = 1:numchunks_local
    
        yi = conv(indata_local(:, ((chunk-1)*maxchunksize+1):min(size(indata_local,2),chunk*maxchunksize) ));

        for i = 1:numlayers
            xi = Wu_local{i}*yi + repmat(bu_local{i}, [1 size(yi,2)]);

            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yi = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            elseif strcmp(layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
                tmp = [];
            elseif strcmp( layertypes{i}, 'softplus' )
                yi = softplus(xi);
            end

        end

        outc = conv(outdata_local(:, ((chunk-1)*maxchunksize+1):min(size(indata_local,2),chunk*maxchunksize) ));
        
        if rms
            
            ll_local = ll_local + makeDouble( -sum(sum((outc - yi).^2)) );
            
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                
                %outc==0 and outc==1 are included in this formula to avoid
                %the annoying case where you have 0*log(0) = 0*-Inf = NaN
                %ll_local = ll_local + makeDouble( sum(sum(outc.*log(yi + (outc==0)) + (1-outc).*log(1-yi + (outc==1)))) );
                
                %this version is more stable:
                ll_local = ll_local + makeDouble(sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0))))));
                
                
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                
                ll_local = ll_local + makeDouble(sum(sum(outc.*log(yi))));
                
                
            elseif strcmp( layertypes{numlayers}, 'linear' )
                
                ll_local = ll_local + 0.5*makeDouble( -sum(sum((outc - yi).^2)) );
                
            end
        end
        xi = [];

        if strcmp( errtype, 'class' )
            %err_local = 1 - makeDouble(sum( sum(outc.*yi,1) == max(yi,[],1) ) );
            err_local = err_local + makeDouble(sum( sum(outc.*yi,1) ~= max(yi,[],1) ) );
        elseif strcmp( errtype, 'L2' )
            err_local = err_local + makeDouble(sum(sum((yi - outc).^2, 1)));
        elseif strcmp( errtype, 'none')
            %do nothing
        else
            error( 'Unrecognized error type' );
        end
        %err_local = makeDouble(   (mones(1,size(in,1))*((yi - out).^2))*mones(size(in,2),1)/conv(size(in,2))  );
        
        outc = [];
        yi = [];
    end

    ll_local = ll_local / size(indata_local,2);
    err_local = err_local / size(indata_local,2);
    
    ll_local = ll_local - 0.5*weightcost*makeDouble(paramsp_local'*(maskp.*paramsp_local));

end



maskp = mones(psize,1);
[maskW, maskb] = unpack(maskp);
for i = 1:length(maskb)
    %maskb{i}(:) = 0; %uncomment this line apply the l_2 only to the connection weights and not the biases
end
%pause
maskp = pack(maskW,maskb);


indata = single(indata);
outdata = single(outdata);
intest = single(intest);
outtest = single(outtest);


function outputString( s )
    fprintf( 1, '%s\n', s );
    fprintf( fid, '%s\r\n', s );
end



fid = fopen( [runName '.txt'], 'a' );

outputString( '' );
outputString( '' );
outputString( '==================== New Run ====================' );
outputString( '' );
outputString( ['Start time: ' datestr(now)] );
outputString( '' );
outputString( ['Description: ' runDesc] );
outputString( '' );


ch = mzeros(psize, 1);
paramsp_avg = mzeros(psize,1);

for i = 1:numlayers

    dEdx1{i} = mzeros( layersizes(i+1), 1 );
    dEdx1_emp{i} = mzeros( layersizes(i+1), 1 );
    y_hom1{i} = mzeros( layersizes(i)+1, 1 );

    dEdx2{i,i} = mzeros( layersizes(i+1) );
    dEdx2_emp{i,i} = mzeros( layersizes(i+1) );
    y_hom2{i,i} = mzeros( layersizes(i)+1 );
    

    %initial values to be used if we are updating inverses using a Newton iteration or something
    inv_dEdx2_damp{i} = mzeros( layersizes(i+1) );
    inv_y_hom2_damp{i} = mzeros( layersizes(i)+1 );

    if i < numlayers
        dEdx2{i,i+1} = mzeros( layersizes(i+1), layersizes(i+2) );
        dEdx2_emp{i,i+1} = mzeros( layersizes(i+1), layersizes(i+2) );
        y_hom2{i,i+1} = mzeros( layersizes(i)+1, layersizes(i+1)+1 );
    end

end




if ~isempty( resumeFile )
    outputString( ['Resuming from file: ' resumeFile] );
    outputString( '' );
    
    load( resumeFile );
    
    ch = conv(ch);
    paramsp = conv(paramsp);
    paramsp_avg = conv(paramsp_avg);

    for i = 1:numlayers
        y_hom2{i,i} = conv(y_hom2{i,i});
        dEdx2{i,i} = conv(dEdx2{i,i});
        if i < numlayers
            y_hom2{i,i+1} = conv(y_hom2{i,i+1});
            dEdx2{i,i+1} = conv(dEdx2{i,i+1});
        end
    end
    
    %legacy compatibility ('iter' used to be named 'epoch')
    if exist('epoch', 'var')
        iter = epoch;
        clear epoch
    end

    iter = iter + 1;
else
    
    lambda = initlambda;
    gamma = initgamma;    
    
    %llrecord = zeros(maxiter,4);
    %errrecord = zeros(maxiter,4);
    %lambdarecord = zeros(maxiter,1);
    %times = zeros(maxiter,1);
    
    llrecord = zeros(0,4);
    errrecord = zeros(0,4);
    lambdarecord = zeros(0,1);
    times = zeros(0,1);

    iter = 1;
    
end

if isempty(paramsp)
    if ~isempty(Win)
        paramsp = pack(Win,bin);
        clear Win bin
    else
        
        
        %SPARSE INIT:
        paramsp = zeros(psize,1); %not mzeros
        
        [Wtmp,btmp] = unpack(paramsp);
        
        numconn = 15;
        
        for i = 1:numlayers
 
            initcoeff = 1.0;

            if i > 1 && strcmp( layertypes{i-1}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            if strcmp( layertypes{i}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            
            if strcmp( layertypes{i}, 'tanh' )
                btmp{i}(:) = 0.5;
            end
            
            if strcmp( layertypes{i}, 'softplus' )
                initcoeff = 0.1*initcoeff;
            end
            
            %outgoing
            %{
            for j = 1:layersizes(i)
                idx = ceil(layersizes(i+1)*rand(1,numconn));
                Wtmp{i}(idx,j) = randn(numconn,1)*coeff;
            end
            %}
            
            %incoming
            for j = 1:layersizes(i+1)
                idx = ceil(layersizes(i)*rand(1,numconn));
                Wtmp{i}(j,idx) = randn(numconn,1)*initcoeff;
            end
        end
        
        
        %{
        %GLOROT-BENGIO INIT:
        
            paramsp = zeros(psize,1);

            [Wtmp,btmp] = unpack(paramsp);
            for i = 1:numlayers

                Wtmp{i} = (2*rand(layersizes(i+1), layersizes(i))-1)*sqrt(6/(1+layersizes(i+1)+layersizes(i)));


                if strcmp( layertypes{i}, 'logistic' )
                    Wtmp{i} = Wtmp{i}*2;
                end

                if i > 1 && strcmp( layertypes{i-1}, 'logistic' )
                    btmp{i} = -1*sum(Wtmp{i}, 2);
                end

                if i > 1 && strcmp( layertypes{i-1}, 'logistic' )
                    Wtmp{i} = Wtmp{i}*2;
                end

                %added by me:
                %btmp{i} = 0.2*(2*rand(layersizes(i+1),1)-1);
            end        
        %}  
        
        
        paramsp = pack(Wtmp, btmp);
        
        clear Wtmp btmp
    end
    
elseif size(paramsp,1) ~= psize || size(paramsp,2) ~= 1
    error( 'Badly sized initial parameter vector.' );
else
    paramsp = conv(paramsp);
end

outputString( 'Initial constant values:' );
outputString( '------------------------' );
outputString( '' );
for i = 1:length(rec_constants)
    outputString( [rec_constants{i} ': ' num2str(eval( rec_constants{i} )) ] );
end

outputString( '' );
outputString( '=================================================' );
outputString( '' );


%set to zero after loading (since these are normally not saved)
if computeOffTriDiag
    for i = 1:numlayers
        for j = i+2:numlayers
            dEdx2{i,j} = mzeros( layersizes(i+1), layersizes(j+1) );
            y_hom2{i,j} = mzeros( layersizes(i)+1, layersizes(j)+1 );
        end
    end
end


grad2 = mzeros(psize,1);


firstIterFlag = 1;

startingIter = iter;

for iter = iter:maxiter

    tic

    
    %for now we always do this:
    updateStats = 1;
    
    
    refreshEIGs = 0;
    refreshInvApprox = 0;

    if mod(iter, refreshInvApprox_interval) == 0 || iter <= 3 || firstIterFlag

        refreshEIGs = 1;
        refreshInvApprox = 1;
        
    end
    
    
    %an exponentially increasing schedule for size_minibatch
    div = (minibatch_maxsize_targetiter-1)/log2( minibatch_maxsize/minibatch_startsize );
    size_minibatch = min( floor( 2^((iter-1)/div )*minibatch_startsize ), minibatch_maxsize);

    %{
    %a linearly increasing schedule for size_minibatch
    slope = (minibatch_maxsize - minibatch_startsize)/minibatch_maxsize_targetiter;
    size_minibatch = min( minibatch_startsize + slope*iter, minibatch_maxsize);
    %}
    
    
    outputString( ['size_minibatch = ' num2str(size_minibatch) ] );
    
    
    
    %idx_minibatch = ceil(rand( size_minibatch, 1 )*numcases); %sample with replacement
    idx_minibatch = randperm(numcases); idx_minibatch = idx_minibatch(1:size_minibatch); %sample without replacement
    
    indata_minibatch = indata(:, idx_minibatch );
    outdata_minibatch = outdata(:, idx_minibatch );

    size_sampleminibatch = ceil(ratio_sample*size_minibatch);
    
    size_BVminibatch = ceil(ratio_BV*size_minibatch);
    
    
    
    sizechunk = [maxchunksize*ones(1,floor(size_minibatch/maxchunksize)) mod(size_minibatch, maxchunksize)];
    if sizechunk(end) == 0
        sizechunk = sizechunk(1:end-1);
    end
    sizechunk_sample = [maxchunksize*ones(1,floor(size_sampleminibatch/maxchunksize)) mod(size_sampleminibatch, maxchunksize)];
    if sizechunk_sample(end) == 0
        sizechunk_sample = sizechunk_sample(1:end-1);
    end
    sizechunk_BV = [maxchunksize*ones(1,floor(size_BVminibatch/maxchunksize)) mod(size_BVminibatch, maxchunksize)];
    if sizechunk_BV(end) == 0
        sizechunk_BV = sizechunk_BV(1:end-1);
    end

    
    
    numchunks = length(sizechunk);
    numchunks_sample = length(sizechunk_sample);
    numchunks_BV = length(sizechunk_BV);

    %disp( ['numchunks = ' num2str(numchunks) ] );
    
    
    
    for i = 1:numlayers

        dEdx1_inc{i} = mzeros( layersizes(i+1), 1 );
        y_hom1_inc{i} = mzeros( layersizes(i)+1, 1 );
        
        dEdx2_inc{i,i} = mzeros( layersizes(i+1) );
        y_hom2_inc{i,i} = mzeros( layersizes(i)+1 );
        
        if i < numlayers && useTriBlock
            dEdx2_inc{i,i+1} = mzeros( layersizes(i+1), layersizes(i+2) );
            y_hom2_inc{i,i+1} = mzeros( layersizes(i)+1, layersizes(i+1)+1 );
        end


        if stochastic_mode == 0
            dEdx1_emp_inc{i} = mzeros( layersizes(i+1), 1 );
            dEdx2_emp_inc{i,i} = mzeros( layersizes(i+1) );
            if i < numlayers && useTriBlock
                dEdx2_emp_inc{i,i+1} = mzeros( layersizes(i+1), layersizes(i+2) );
            end
        end
    end

    
    [Wu, bu] = unpack(paramsp);
    
    ll = 0;
    err = 0;

    grad = mzeros(psize,1);
    if stochastic_mode == 0
        grad2_emp_inc = mzeros(psize,1);
    end
    
    y = cell(numlayers+1,1);

    
    
    for chunk = 1:numchunks

        %forward prop:
        %index transition takes place at nonlinearity

        y{1,chunk} = store(indata_minibatch(:, ((chunk-1)*maxchunksize+1):min(size_minibatch,chunk*maxchunksize) ));

        yip1 = conv( y{1,chunk} );

        dEdW = cell(numlayers, 1);
        dEdb = cell(numlayers, 1);

        dEdW2 = cell(numlayers, 1);
        dEdb2 = cell(numlayers, 1);

        
        
        for i = 1:numlayers

            yi = yip1;
            yip1 = [];
            xi = Wu{i}*yi + repmat(bu{i}, [1 size(yi,2)]);

            %x1{i} = x1{i} + sum(xi,2);
            %x2{i} = x2{i} + xi*xi';


            yi = [];

            if strcmp(layertypes{i}, 'logistic')
                yip1 = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yip1 = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yip1 = xi;
            elseif strcmp( layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yip1 = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );
                tmp = [];
            elseif strcmp( layertypes{i}, 'softplus' ) %a smooth version of "RELUs"
                yip1 = softplus(xi);
            else
                error( 'Unknown/unsupported layer type' );
            end

            y{i+1,chunk} = store(yip1);
        end

        %back prop:
        outc = conv(outdata_minibatch(:, ((chunk-1)*maxchunksize+1):min(size_minibatch,chunk*maxchunksize) ));

        if rms
            ll = ll + makeDouble( -sum(sum((outc - yip1).^2)) );
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                %ll = ll + makeDouble( sum(sum(outc.*log(yip1 + (outc==0)) + (1-outc).*log(1-yip1 + (outc==1)))) );
                %more stable:
                ll = ll + makeDouble(sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0))))));
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                ll = ll + makeDouble(sum(sum(outc.*log(yip1))));
            elseif strcmp( layertypes{numlayers}, 'linear' )
                ll = ll + 0.5*makeDouble( -sum(sum((outc - yip1).^2)) );
            end
        end
        xi = [];

        
        if strcmp( errtype, 'class' )
            %err = 1 - makeDouble(sum( sum(outc.*yi,1) == max(yi,[],1) ) );
            err = err + makeDouble(sum( sum(outc.*yip1,1) ~= max(yip1,[],1) ) );
        elseif strcmp( errtype, 'L2' )
            err = err + makeDouble(sum(sum((yip1 - outc).^2, 1)));
        elseif strcmp( errtype, 'none')
            %do nothing
        else
            error( 'Unrecognized error type' );
        end



        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    dEdxi = dEdyip1.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    dEdxi = dEdyip1;
                elseif strcmp( layertypes{i}, 'softplus' )
                    dEdxi = dEdyip1.*dsoftplus(yip1);
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    dEdxi = outc - yip1; %simplified due to canonical link
                else
                    dEdyip1 = 2*(outc - yip1);

                    if strcmp( layertypes{i}, 'softmax' )
                        dEdxi = dEdyip1.*yip1 - yip1.* repmat( sum( dEdyip1.*yip1, 1 ), [layersizes(i+1) 1] );
                        %error( 'RMS error not supported with softmax output' );

                    elseif strcmp(layertypes{i}, 'logistic')
                        dEdxi = dEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        dEdxi = dEdyip1;
                    elseif strcmp( layertypes{i}, 'softplus' )
                        dEdxi = dEdyip1.*dsoftplus(yip1);
                    else
                        error( 'Unknown/unsupported layer type' );
                    end

                    dEdyip1 = [];

                end
                outc = [];
            end

            %yip1 = [];


            dEdyi = Wu{i}'*dEdxi;

            yi = conv(y{i,chunk});

            %standard gradient comp:
            dEdW{i} = dEdxi*yi';
            dEdb{i} = sum(dEdxi,2);
            
            if storeD
                dEdx_emp{i,chunk} = store(dEdxi); %don't really need this except for compute_invBlkV_vec and HV products
                dEdy_emp{i,chunk} = store(dEdyi);
            end            
            
            if updateStats
                if stochastic_mode == 0 && chunk <= numchunks_sample
                    dEdxi_sub = dEdxi(:,1:sizechunk_sample(chunk));

                    dEdx1_emp_inc{i} = dEdx1_emp_inc{i} + sum(dEdxi_sub,2) / size_sampleminibatch;
                    dEdx2_emp_inc{i,i} = dEdx2_emp_inc{i,i} + (dEdxi_sub*dEdxi_sub') / size_sampleminibatch;


                    if i < numlayers && useTriBlock
                        dEdxip1_sub = dEdxip1(:,1:sizechunk_sample(chunk));

                        dEdx2_emp_inc{i,i+1} = dEdx2_emp_inc{i,i+1} + (dEdxi_sub*dEdxip1_sub') / size_sampleminibatch;
                        dEdxip1_sub = [];
                    end

                end

                if chunk <= numchunks_sample
                    yi_hom = [yi(:,1:sizechunk_sample(chunk)) ; mones( 1, sizechunk_sample(chunk) )];

                    y_hom1_inc{i} = y_hom1_inc{i} + sum(yi_hom,2) / conv(size_sampleminibatch);
                    y_hom2_inc{i,i} = y_hom2_inc{i,i} + (yi_hom*yi_hom') / conv(size_sampleminibatch);

                    if i < numlayers && useTriBlock
                        yip1_hom = [yip1(:,1:sizechunk_sample(chunk)) ; mones( 1, sizechunk_sample(chunk) )];

                        y_hom2_inc{i,i+1} = y_hom2_inc{i,i+1} + (yi_hom*yip1_hom') / conv(size_sampleminibatch);
                        yip1_hom = [];
                    end
                end

                if stochastic_mode == 0

                    if computeGrad2
                        tmp = (dEdxi_sub.^2)*(yi_hom.^2)';
                        dEdW2{i} = tmp(:,1:end-1);
                        dEdb2{i} = tmp(:,end);
                    end

                    dEdxi_sub = [];
                end
            end
            
            dEdxip1 = dEdxi;

            yi_hom = [];
            
            dEdxi = [];

            dEdyip1 = dEdyi;
            dEdyi = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];  dEdyip1 = [];  dEdxip1 = [];


        grad = grad + pack(dEdW, dEdb);

        if stochastic_mode == 0 && computeGrad2
            grad2_emp_inc = grad2_emp_inc + pack(dEdW2, dEdb2);
        end

        dEdW = []; dEdb = []; dEdW2 = []; dEdb2 = [];

    end
    
    grad = grad / conv(size_minibatch);
    grad = grad - conv(weightcost)*(maskp.*paramsp);

    if stochastic_mode == 0 && computeGrad2
        grad2_emp_inc = grad2_emp_inc / conv(size_sampleminibatch);
    end

    
    
    ll = ll / size_minibatch;
    err = err / size_minibatch;
    ll = ll - 0.5*weightcost*makeDouble(paramsp'*(maskp.*paramsp));
    
    oldll = ll;
    olderr = err;
    ll = [];
    err = [];
    
    
    %gtmp = gradcomp(@(x)computeLL(x,indata_minibatch, outdata_minibatch), paramsp, 1e-6); %gradient check (doubles mode)

    if updateStats
        
        if stochastic_mode == 0

            %Use training distribution (gives approx of emprical Fisher)
            
            dEdx1_inc = dEdx1_emp_inc;
            dEdx2_inc = dEdx2_emp_inc;
            grad2_inc = grad2_emp_inc;

        else

            %Use model's distribution (gives approx of proper Fisher).  For this we need to do extra backwards passes, which are performed below
            
            puregrad = mzeros(psize,1);
            grad2_inc = mzeros(psize,1);


            for chunk = 1:numchunks_sample

                for samp = 1:numsamples

                    dEdW = cell(numlayers, 1);
                    dEdb = cell(numlayers, 1);

                    dEdW2 = cell(numlayers, 1);
                    dEdb2 = cell(numlayers, 1);

                    %back prop:

                    yip1 = conv(y{numlayers+1,chunk}(:,1:sizechunk_sample(chunk)));

                    for i = numlayers:-1:1

                        if i < numlayers
                            %logistics:
                            if strcmp(layertypes{i}, 'logistic')
                                dEdxi = dEdyip1.*yip1.*(1-yip1);
                            elseif strcmp(layertypes{i}, 'tanh')
                                dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                            elseif strcmp(layertypes{i}, 'linear')
                                dEdxi = dEdyip1;
                            elseif strcmp( layertypes{i}, 'softplus' )
                                dEdxi = dEdyip1.*dsoftplus(yip1);
                            else
                                error( 'Unknown/unsupported layer type' );
                            end
                        else
                            if ~rms
                                %we randomly sample the targets

                                %dEdxi = outc - yip1; %simplified due to canonical link

                                if strcmp(layertypes{i}, 'logistic')
                                    dEdxi = (yip1 >= mrand(size(yip1))) - yip1;
                                elseif strcmp(layertypes{i}, 'softmax')
                                    %error('not implemented yet');

                                    dEdxi = samplesoftmax(yip1) - yip1;
                                elseif strcmp(layertypes{i}, 'linear')
                                    dEdxi = mrandn(size(yip1));
                                else
                                    error('not implemented');
                                end

                            else
                                dEdyip1 = sqrt(2)*mrandn(size(yip1));

                                if strcmp( layertypes{i}, 'softmax' )
                                    dEdxi = dEdyip1.*yip1 - yip1.*repmat( sum( dEdyip1.*yip1, 1 ), [layersizes(i+1) 1] );
                                    %error( 'RMS error not supported with softmax output' );

                                elseif strcmp(layertypes{i}, 'logistic')
                                    dEdxi = dEdyip1.*yip1.*(1-yip1);
                                elseif strcmp(layertypes{i}, 'tanh')
                                    dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                                elseif strcmp(layertypes{i}, 'linear')
                                    dEdxi = dEdyip1;
                                else
                                    error( 'Unknown/unsupported layer type' );
                                end

                                dEdyip1 = [];

                            end

                            outc = [];

                        end

                        if computeOffTriDiag
                            %storing these only for off-tridiagonal computations:
                            dEdx{i,chunk,samp} = store(dEdxi);
                        end

                        dEdyi = Wu{i}'*dEdxi;

                        yi = conv(y{i,chunk}(:,1:sizechunk_sample(chunk)));

                        if computeGrad2
                            dEdW2{i} = (dEdxi.^2)*(yi.^2)';
                            dEdb2{i} = sum(dEdxi.^2,2);
                        end

                        dEdx1_inc{i} = dEdx1_inc{i} + sum(dEdxi,2) / conv(size_sampleminibatch*numsamples);
                        dEdx2_inc{i,i} = dEdx2_inc{i,i} + (dEdxi*dEdxi') / conv(size_sampleminibatch*numsamples);

                        if i < numlayers && useTriBlock
                            dEdx2_inc{i,i+1} = dEdx2_inc{i,i+1} + (dEdxi*dEdxip1') / conv(size_sampleminibatch*numsamples);
                        end

                        dEdxip1 = dEdxi;

                        dEdxi = [];

                        dEdyip1 = dEdyi;
                        dEdyi = [];

                        yip1 = yi;
                        yi = [];
                    end
                    yip1 = [];  dEdyip1 = [];  dEdxip1 = [];

                    if computeGrad2
                        grad2_inc = grad2_inc + pack(dEdW2, dEdb2) / conv(size_sampleminibatch*numsamples);
                    end

                    dEdW = []; dEdb = []; dEdW2 = []; dEdb2 = [];

                end

            end

        end
    end

    
    if computeOffTriDiag
        %compute extra off-tridiagonal statistics (used in some testing routines):

        if stochastic_mode == 0 %|| numsamples > 1
            error('not implemented');
        end
        
        for i = 1:numlayers
            for j = i+2:numlayers
                dEdx2_inc{i,j} = mzeros( layersizes(i+1), layersizes(j+1) );
                y_hom2_inc{i,j} = mzeros( layersizes(i)+1, layersizes(j)+1 );
            end
        end
        
        for chunk = 1:numchunks_sample
            for i = 1:numlayers

                dEdxi = conv(dEdx{i,chunk,samp});
                
                yi = conv(y{i,chunk}(:,1:sizechunk_sample(chunk)));
                yi_hom = [yi; mones( 1, size(yi,2) )];

                for j = i+2:numlayers
                    dEdxj = conv(dEdx{j,chunk,samp});

                    dEdx2_inc{i,j} = dEdx2_inc{i,j} + (dEdxi*dEdxj') / conv(size_sampleminibatch);

                    yj = conv(y{j,chunk}(:,1:sizechunk_sample(chunk)));
                    yj_hom = [yj; mones( 1, size(yj,2) )];

                    y_hom2_inc{i,j} = y_hom2_inc{i,j} + (yi_hom*yj_hom') / conv(size_sampleminibatch);

                end
            end
        end
            
    end


    if updateStats
       
       fade = min( 1 - 1/(iter-startingIter+1), maxfade_stats );
       
       %outputString( ['fade = ' num2str(fade)] );


       fadein = 1-fade;

        for i = 1:numlayers
            dEdx1{i} = conv(fade)*dEdx1{i} + conv(fadein)*dEdx1_inc{i};
            dEdx2{i,i} = conv(fade)*dEdx2{i,i} + conv(fadein)*dEdx2_inc{i,i};

            y_hom1{i} = conv(fade)*y_hom1{i} + conv(fadein)*y_hom1_inc{i};
            y_hom2{i,i} = conv(fade)*y_hom2{i,i} + conv(fadein)*y_hom2_inc{i,i};

            if i < numlayers && useTriBlock
                dEdx2{i,i+1} = conv(fade)*dEdx2{i,i+1} + conv(fadein)*dEdx2_inc{i,i+1};
                y_hom2{i,i+1} = conv(fade)*y_hom2{i,i+1} + conv(fadein)*y_hom2_inc{i,i+1};
            end

            dEdx2{i,i} = (dEdx2{i,i} + dEdx2{i,i}')/2; 
            y_hom2{i,i} = (y_hom2{i,i} + y_hom2{i,i}')/2; 

        end

        if computeOffTriDiag
            for i = 1:numlayers
                for j = i+2:numlayers
                    dEdx2{i,j} = conv(fade)*dEdx2{i,j} + conv(fadein)*dEdx2_inc{i,j};
                    y_hom2{i,j} = conv(fade)*y_hom2{i,j} + conv(fadein)*y_hom2_inc{i,j};
                end
            end
        end

        grad2 = conv(fade)*grad2 + conv(fadein)*grad2_inc;
    end

    
       
    if useEIGs && refreshEIGs
        for i = 1:numlayers
            %pre-compute the eigen-decompositions
            [U_dEdx2{i}, tmp] = meig( dEdx2{i,i} ); D_dEdx2{i} = diag(tmp);
            [U_y_hom2{i}, tmp] = meig( y_hom2{i,i} ); D_y_hom2{i} = diag(tmp);
        end
    end
    

    if mod(iter, gamma_adj_interval) == 0
        gammas = [ max( (gamma_drop^gamma_adj_interval)*gamma, gamma_min ) gamma min( (gamma_boost^gamma_adj_interval)*gamma, gamma_max )]; %the range of new gamma values to try
    else
        gammas = [gamma];
    end
    
    
    oldch = ch;

    vals = [];
    lls = [];
    

    
    for gamma_idx = 1:length(gammas)

        gamma = gammas(gamma_idx);
        
        if refreshInvApprox
            
            if useEIGs
            
                for i = 1:numlayers

                    %load pre-computed versions from above loop:
                    D_dEdxi2 = D_dEdx2{i};
                    D_yi_hom2 = D_y_hom2{i};
                    

                    %Various different options for computing the "pi" ratio from the paper:
                    %ratio = sqrt(median(D_yi_hom2)/median(D_dEdxi2));
                    %ratio = sqrt(max(D_yi_hom2)/max(D_dEdxi2));
                    ratio = sqrt(sum(D_yi_hom2)*length(D_dEdxi2)/(sum(D_dEdxi2)*length(D_yi_hom2)));  %nuclear/trace-norm (Also very good.  And for convnets too)
                    %ratio = sqrt( (sqrt(length(D_dEdxi2))*norm(D_yi_hom2))/(sqrt(length(D_yi_hom2))*norm(D_dEdxi2)) ); %used in paper
                    %ratio = 1;

                    coeff1 = conv(gamma)/ratio;

                    D_dEdxi2 = D_dEdxi2 + coeff1;
                    dEdx2_damp{i,i} = dEdx2{i,i} + coeff1*meye( size(dEdx2{i,i}) );


                    coeff2 = conv(gamma)*ratio;

                    D_yi_hom2 = D_yi_hom2 + coeff2;
                    y_hom2_damp{i,i} = y_hom2{i,i} + coeff2*meye( size(y_hom2{i,i}) );



                    %inv_dEdx2_damp{i} = U_dEdx2{i}*diag(1./D_dEdxi2)*U_dEdx2{i}';
                    inv_dEdx2_damp{i} = U_dEdx2{i}*(repmat(1./D_dEdxi2, [1 size(U_dEdx2{i},1)]).*U_dEdx2{i}');
                    %inv_y_hom2_damp{i} = U_y_hom2{i}*diag(1./D_yi_hom2)*U_y_hom2{i}';
                    inv_y_hom2_damp{i} = U_y_hom2{i}*(repmat(1./D_yi_hom2, [1 size(U_y_hom2{i},1)]).*U_y_hom2{i}');

                    if useTriBlock
                        %inv_half_dEdx2_damp{i} = U_dEdx2{i}*diag(1./sqrt(D_dEdxi2))*U_dEdx2{i}';
                        inv_half_dEdx2_damp{i} = U_dEdx2{i}*(repmat(1./sqrt(D_dEdxi2), [1 size(U_dEdx2{i},1)]).*U_dEdx2{i}');
                        %inv_half_y_hom2_damp{i} = U_y_hom2{i}*diag(1./sqrt(D_yi_hom2))*U_y_hom2{i}';
                        inv_half_y_hom2_damp{i} = U_y_hom2{i}*(repmat(1./sqrt(D_yi_hom2), [1 size(U_y_hom2{i},1)]).*U_y_hom2{i}');
                    end

                    %inv_half_dEdx2_damp{i} = mchol(inv_dEdx2_damp{i})';
                    %inv_half_y_hom2_damp{i} = mchol(inv_y_hom2_damp{i})';

                    %{
                    %only needed for blkCholTriBlk stuff:
                    half_dEdx2{i} = U_dEdx2{i}*diag(sqrt(D_dEdxi2))*U_dEdx2{i}';
                    half_y_hom2{i} = U_y_hom2{i}*diag(sqrt(D_yi_hom2))*U_y_hom2{i}';

                    %half_dEdx2{i} = minv(inv_half_dEdx2_damp{i});
                    %half_y_hom2{i} = minv(inv_half_y_hom2_damp{i});
                    %}

                end
                
            else
                %Direct inversion might be more efficient in some settings
                for i = 1:numlayers
                    
                    %Various different options for computing the "pi" ratio from the paper:
                    ratio = sqrt(trace(y_hom2{i})*length(D_dEdxi2)/(trace(dEdx2{i})*length(D_yi_hom2))); %nuclear/trace-norm (Also very good.  And for convnets too)
                    %ratio = sqrt( (sqrt(size( dEdx2{i,i}, 2 ))*norm( y_hom2{i,i}, 'fro' ))/(sqrt(size( y_hom2{i,i}, 2 ))*norm( dEdx2{i,i}, 'fro' )) ); %used in paper
                    %ratio = 1;

                    coeff1 = conv(gamma)/ratio;
                    coeff2 = conv(gamma)*ratio;

                    dEdx2_damp{i,i} = dEdx2{i,i} + coeff1*meye( size(dEdx2{i,i}) );
                    inv_dEdx2_damp{i} = minv(dEdx2_damp{i,i});

                    y_hom2_damp{i,i} = y_hom2{i,i} + coeff2*meye( size(y_hom2{i,i}) );
                    inv_y_hom2_damp{i} = minv(y_hom2_damp{i,i});

                    if useTriBlock
                        inv_half_dEdx2_damp{i} = mchol(inv_dEdx2_damp{i})';
                        inv_half_y_hom2_damp{i} = mchol(inv_y_hom2_damp{i})';
                    end

                end
            end


            if useTriBlock
                
                %Pre-compute various quantities used by compute_invTriBlkV

                for i = 2:numlayers
                    predweights_dEdx{i} = dEdx2{i-1,i}*inv_dEdx2_damp{i};
                    predweights_y_hom{i} = y_hom2{i-1,i}*inv_y_hom2_damp{i};
                end

                for i = 1:numlayers-1

                    tmp = dEdx2{i,i+1}'*inv_half_dEdx2_damp{i}; M_dEdx{i} = tmp'*inv_dEdx2_damp{i+1}*tmp; %slightly faster version of above
                    %M_dEdx{i} = inv_half_dEdx2_damp{i}'*(dEdx2{i,i+1}*inv_dEdx2_damp{i+1}*dEdx2{i,i+1}')*inv_half_dEdx2_damp{i};
                    M_dEdx{i} = (M_dEdx{i} + M_dEdx{i}')/2;  %fix numerical problems leading to non-orthogonal U's with complex entries
                    [UM_dEdx{i},tmp] = meig(M_dEdx{i}); DM_dEdx{i} = diag(tmp);

                    K_dEdx{i} = inv_half_dEdx2_damp{i}*UM_dEdx{i};


                    tmp = y_hom2{i,i+1}'*inv_half_y_hom2_damp{i}; M_y_hom{i} = tmp'*inv_y_hom2_damp{i+1}*tmp; %slightly faster version of above
                    %M_y_hom{i} = inv_half_y_hom2_damp{i}'*(y_hom2{i,i+1}*inv_y_hom2_damp{i+1}*y_hom2{i,i+1}')*inv_half_y_hom2_damp{i};
                    M_y_hom{i} = (M_y_hom{i} + M_y_hom{i}')/2;  %fix numerical problems leading to non-orthogonal U's with complex entries
                    [UM_y_hom{i},tmp] = meig(M_y_hom{i}); DM_y_hom{i} = diag(tmp);

                    K_y_hom{i} = inv_half_y_hom2_damp{i}*UM_y_hom{i};


                    %These are supposed to be max 1 mathematically, but this code enforces it numerically:
                    DM_dEdx{i} = min(DM_dEdx{i}, 1.0);
                    DM_y_hom{i} = min(DM_y_hom{i}, 1.0);
                    
                    %{
                    %extra quantities for more efficient version of compute_triBlkV (disable in general!)
                    %Z_dEdx{i} = dEdx2{i,i+1}*inv_dEdx2_damp{i+1}*dEdx2{i,i+1}';
                    Z_dEdx{i} = predweights_dEdx{i+1}*dEdx2{i,i+1}';

                    %Z_y_hom{i} = y_hom2{i,i+1}*inv_y_hom2_damp{i+1}*y_hom2{i,i+1}'
                    Z_y_hom{i} = predweights_y_hom{i+1}*y_hom2{i,i+1}';
                    %}
                    
                end
            end
            
            inv_dEdx2_damp_storage{gamma_idx} = inv_dEdx2_damp;
            inv_y_hom2_damp_storage{gamma_idx} = inv_y_hom2_damp;

            if useTriBlock
                K_dEdx_storage{gamma_idx} = K_dEdx;
                K_y_hom_storage{gamma_idx} = K_y_hom;
                
                predweights_dEdx_storage{gamma_idx} = predweights_dEdx;
                predweights_y_hom_storage{gamma_idx} = predweights_y_hom;
                
                DM_dEdx_storage{gamma_idx} = DM_dEdx;
                DM_y_hom_storage{gamma_idx} = DM_y_hom;
            end            
                
        end


        if useTriBlock
            preconFunc = @compute_invTriBlkV;
            %invpreconFunc = @compute_triBlkV;
        else
            preconFunc = @compute_invBlkV;
            %invpreconFunc = @compute_BlkV;
        end

        %Use the quadratic model derived from the exact Fisher to compute an effective learning rate and momentum decay constant
        if useMomentum
            v = {preconFunc(grad), oldch};
        else
            v = {preconFunc(grad)};
        end
        [chs{gamma_idx}, vals(gamma_idx), coeffs{gamma_idx}] = basic_quad_opt1_factored7( computeBfactV, grad, v, zeros(psize,1), numchunks_BV, makeDouble );
        
        

        %Fixed learning rate (alpha) and momentum decay (mu) as alternative to determining these from quadratic model (using basic_quad_opt1_factored6 as above).  Would need to tune these, and turn off gamma and lambda adjustment (i.e. setting lambda/gamma_adj_interval = Inf), etc.
        %{
        alpha = 0.01; mu = 0.95;
        if ~useMomentum
            mu = 0;
        end
        chs{gamma_idx,lambda_idx} = alpha*preconFunc(grad) + mu*oldch;
        vals(gamma_idx,lambda_idx) = 0; %we don't have a value for this
        coeffs{gamma_idx,lambda_idx} = [alpha mu];
        steps = 1;
        %}
        
    end
    
    [val, gamma_idx_best] = min( vals );
    
    val = vals(gamma_idx_best);
    gamma = gammas(gamma_idx_best);
    ch = chs{gamma_idx_best};
    coeff = coeffs{gamma_idx_best};
    
    
    if refreshInvApprox
        inv_dEdx2_damp = inv_dEdx2_damp_storage{gamma_idx_best};
        inv_y_hom2_damp = inv_y_hom2_damp_storage{gamma_idx_best};

        if useTriBlock
            K_dEdx = K_dEdx_storage{gamma_idx_best};
            K_y_hom = K_y_hom_storage{gamma_idx_best};

            predweights_dEdx = predweights_dEdx_storage{gamma_idx_best};
            predweights_y_hom = predweights_y_hom_storage{gamma_idx_best};

            DM_dEdx = DM_dEdx_storage{gamma_idx_best};
            DM_y_hom = DM_y_hom_storage{gamma_idx_best};
        end
    end
    
    
    if length(coeff) == 1
        outputString( [ 'alpha: ' num2str(coeff(1)) ] );
    else
        outputString( [ 'alpha: ' num2str(coeff(1)) ', mu: ' num2str(coeff(2)) ] );
    end
    
    
    if length( gammas ) > 1
        outputString(['New gamma: ' num2str(gamma)]);
    end
    
    outputString( ['norm(ch) = ' num2str(norm(ch))] );
    
    
    if mod(iter,lambda_adj_interval) == 0
        %Adjust lambda according to the LM rule

        denom = -val;

        [ll, err] = computeLL(paramsp + ch, indata_minibatch, outdata_minibatch);
        
        rho = (ll-oldll)/denom;
        if oldll - ll > 0
            rho = -Inf;
        end

        outputString( ['rho = ' num2str(rho)] );
        
        if autodamp
            if rho < 0.25 || isnan(rho)
                lambda = min( lambda*(boost^lambda_adj_interval), lambda_max );
            elseif rho > 0.75
                lambda = max( lambda*(drop^lambda_adj_interval), lambda_min );
            end
            outputString(['New lambda: ' num2str(lambda)]);
        end
    end
    
    
    rate = 1.0;
    %Parameter update:
    paramsp = paramsp + conv(rate)*ch;
    
    
    if iter < avg_start
        avgfade = 0.0;
    else
        avgfade = min( 1 - 1/(iter-avg_start+1), maxfade_avg );
    end
        
    avgfadein = 1-avgfade;
    
    paramsp_avg = conv(avgfade)*paramsp_avg + conv(avgfadein)*paramsp;
    

    lambdarecord(iter,1) = lambda;

    %outputString( ['iter: ' num2str(iter) ', Minibatch log likelihood: ' num2str(ll) ', error rate: ' num2str(err) ] );
    if ~isempty(ll)
        outputString( ['iter: ' num2str(iter) ', Minibatch log likelihood (pre-update): ' num2str(oldll) ', error rate (pre-update): ' num2str(olderr) ] );
        outputString( ['iter: ' num2str(iter) ', Minibatch log likelihood (post-update): ' num2str(ll) ', error rate (post-update): ' num2str(err) ] );
    else
        outputString( ['iter: ' num2str(iter) ', Minibatch log likelihood (pre-update): ' num2str(oldll) ', error rate (pre-update): ' num2str(olderr) ] );
    end
    
    
    if strcmp( compMode, 'jacket' )
        gforce %no cheating!
    end
    
    times(iter) = toc;
    %disp(['Time used: ' num2str(times(iter))]);
    
    [ll, err] = computeLL(paramsp, indata, outdata);
    llrecord(iter,1) = ll;
    errrecord(iter,1) = err;

    [ll_test, err_test] = computeLL(paramsp, intest, outtest);
    llrecord(iter,2) = ll_test;
    errrecord(iter,2) = err_test;

    [ll_avg, err_avg] = computeLL(paramsp_avg, indata, outdata);
    llrecord(iter,3) = ll_avg;
    errrecord(iter,3) = err_avg;
    
    [ll_test_avg, err_test_avg] = computeLL(paramsp_avg, intest, outtest);
    llrecord(iter,4) = ll_test_avg;
    errrecord(iter,4) = err_test_avg;
    
    if mod(iter, report_full_obj_every_iter) == 0

        outputString( '---' );
        
        outputString( ['Full log-likelihood: ' num2str(ll) ', error rate: ' num2str(err) ] );
        
        outputString( ['TEST log-likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test) ] );

        outputString( ['Error difference (test - train): ' num2str(err_test - err)] );
        
        if reportAveragedParams
        
            outputString( '---' );
            
            outputString( ['Full log-likelihood (averaged params): ' num2str(ll_avg) ', error rate (averaged params): ' num2str(err_avg) ] );

            outputString( ['TEST log-likelihood (averaged params): ' num2str(ll_test_avg) ', error rate (averaged params): ' num2str(err_test_avg) ] );
            
            outputString( ['Error difference (test - train) (averaged params): ' num2str(err_test_avg - err_avg)] );
            
        end
    end
        
        
    outputString( '' );

    pause(0)
    drawnow
    
    if mod(iter, save_every_iter) == 0
        
        paramsp_tmp = paramsp;
        paramsp = single(paramsp);
        ch_tmp = ch;
        ch = single(ch);

        y_hom2_tmp = y_hom2;
        dEdx2_tmp = dEdx2;
        
        for i = 1:numlayers
            y_hom2{i,i} = single(y_hom2{i,i});
            dEdx2{i,i} = single(dEdx2{i,i});
            if i < numlayers
                y_hom2{i,i+1} = single(y_hom2{i,i+1});
                dEdx2{i,i+1} = single(dEdx2{i,i+1});
            end
        end
        
        paramsp_avg_tmp = paramsp_avg;
        paramsp_avg = single(paramsp_avg);
        
        
        save( [runName '_nnet_running.mat'], 'paramsp', 'ch', 'iter', 'lambda', 'llrecord', 'times', 'errrecord', 'lambdarecord', 'y_hom2', 'dEdx2', 'paramsp_avg', 'gamma' );

        if mod(iter, checkpoint_every_iter) == 0
            save( [runName '_nnet_iter' num2str(iter) '.mat'], 'paramsp', 'ch', 'iter', 'lambda', 'llrecord', 'times', 'errrecord', 'lambdarecord', 'y_hom2', 'dEdx2', 'paramsp_avg', 'gamma' );
        end
        
        paramsp = paramsp_tmp;
        ch = ch_tmp;
        y_hom2 = y_hom2_tmp;
        dEdx2 = dEdx2_tmp;
        paramsp_avg = paramsp_avg_tmp;

        clear paramsp_tmp ch_tmp y_hom2_tmp dEdx2_tmp paramsp_avg_tmp
    end

    firstIterFlag = 0;
end

paramsp = makeDouble(paramsp);

outputString( ['Total time: ' num2str(sum(times)) ] );

fclose(fid);

end
