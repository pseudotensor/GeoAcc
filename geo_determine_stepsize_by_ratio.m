function [x, val, coeff] = geo_determine_stepsize_by_ratio( Afunc_fact, b, v, x0, numchunks, makeDouble )
% v = {Delta1, Delta2}

d = length(v);
assert(d == 2);
%coeff = pinv(M)*c; 
norm_Delta1 = norm(v{1});
norm_Delta2 = norm(v{2});
alpha = norm_Delta1 / norm_Delta2;
coeff = [alpha, -1/2*alpha^2];

x = x0;
for i = 1:d
    x = x + coeff(i)*v{i};
end

disp(coeff)

fprintf('Delta1 norm: %f\n', norm(coeff(1)*v{1}));
fprintf('Delta2 norm: %f\n', norm(coeff(2)*v{2}));

%val = -0.5*(c'*coeff) + val0;
val = 0.5*makeDouble( Afunc_fact(x, chunk)'*Afunc_fact(x, chunk) ) - makeDouble( b'*x );
end
