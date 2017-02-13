function [x, val, coeff] = basic_quad_opt1_factored7( Afunc_fact, b, v, x0, numchunks, makeDouble )
% v = {Delta1, Delta2}

d = length(v);
assert(d == 2);

d = 1;
nonZeroX0 = 0;
if norm(x0) > 0
    nonZeroX0 = 1;
end

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


coeff = pinv(M)*c; 

coeff = [coeff, -1/2*coeff^2];
d = 2;

x = x0;
for i = 1:d
    x = x + coeff(i)*v{i};
end

val0 = 0.5*x0Ax0; 
if nonZeroX0
    val0 = val0 - makeDouble( b'*x0 );
end

disp(coeff)

fprintf('Delta1 norm: %f\n', norm(coeff(1)*v{1}));
fprintf('Delta2 norm: %f\n', norm(coeff(2)*v{2}));

%val = -0.5*(c'*coeff) + val0;
val = 0.5*makeDouble( Afunc_fact(x, chunk)'*Afunc_fact(x, chunk) ) - makeDouble( b'*x )

