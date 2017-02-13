function [x, val, coeff] = geo_quad_opt_momentum( Afunc_fact, b, v, x0, numchunks, makeDouble )
% v = {Delta1, Delta2, oldch}

function [y, grady] = quadobj(x,Q,f,c)
    y = 1/2*x'*Q*x + f'*x + c;
    if nargout > 1
        grady = Q*x + f;
    end
end

function [y,yeq] = quadconstr(x,H,k,d)    
    yeq = 1/2*x'*H*x + k'*x + d;

    y = [];
end

function hess = quadhess(x,lambda,Q,H)
    hess = Q;
    jj = length(H); % jj is the number of inequality constraints
    for i = 1:jj
        hess = hess + lambda.eqnonlin(i)*H{i};
    end
end


d = length(v);

assert(d == 3);

nonZeroX0 = 0;
if norm(x0) > 0
    nonZeroX0 = 1;
end

%strip away 0 vectors
j = 0;
for i = 1:d
    if norm(v{i}) > 0
        j = j + 1;
        v_new{j} = v{i};
    end
end
d = j;
v = v_new;

c = zeros(d,1);

for i = 1:d
    c(i) = makeDouble( b'*v{i} );
end


M = zeros(d,d);

x0Ax0 = 0;

for chunk = 0:numchunks

    A_fact_v = cell(1,d);
    
    for i = 1:d
        A_fact_v{i} = Afunc_fact(v{i},chunk);
    end
    
    for i = 1:d
        for j = i:d
            M(i,j) = M(i,j) + makeDouble( A_fact_v{i}'*A_fact_v{j} );
        end
    end
    
    if nonZeroX0
        A_fact_x0 = Afunc_fact(x0,chunk);
        
        for i = 1:d
            c(i) = c(i) - makeDouble( A_fact_x0'*A_fact_v{i} );
        end
        
        x0Ax0 = x0Ax0 + makeDouble( A_fact_x0'*A_fact_x0 );
    end
    
end

for i = 1:d
    for j = i+1:d
        M(j,i) = M(i,j);
    end
end


fun = @(x)quadobj(x,M,-c,0);
H = zeros(d,d);
H(1,1) = 1;
k = zeros(d,1);
k(2) = 1;
nonlconstr = @(x)quadconstr(x,H,k,0);
init_x= zeros(d,1); % column vector

options = optimoptions(@fmincon,'Algorithm','interior-point', 'GradObj', 'on');


[coeff,fval,eflag,output,lambda] = fmincon(fun,init_x,...
    [],[],[],[],[],[],nonlconstr,options);

%coeff = pinv(M)*c; 

x = x0;
for i = 1:d
    x = x + coeff(i)*v{i};
end

val0 = 0.5*x0Ax0; 
if nonZeroX0
    val0 = val0 - makeDouble( b'*x0 );
end

%val = -0.5*(c'*coeff) + val0;
val = 0.5*makeDouble( Afunc_fact(x, chunk)'*Afunc_fact(x, chunk) ) - makeDouble( b'*x );
end
