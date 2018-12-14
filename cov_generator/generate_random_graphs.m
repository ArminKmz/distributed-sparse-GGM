% name = 'synthetic data';
% save('random_covs.mat', 'name');

for p=5:5:100
   k=1;
   disp(p);
   while k <= 20
       [Q_inv] = randomInvCovGenerator(p, .1, 5);
%        disp(min(Q_inv(Q_inv~=0)));
       if check(Q_inv)
%           name = strcat('Qinv_', int2str(p), '_', num2str(k));
%           S.(name) = Q_inv;
%           save('random_covs.mat', '-struct', 'S', (name), '-append');
          k = k + 1;
       end
   end
end


% testing check.m
% rho = .1;
% Q = [1 rho rho rho; rho 1 rho^2 rho^2; rho rho^2 1 rho^2; rho rho^2 rho^2 1];
% Q_inv = inv(Q);
% Q_inv(abs(Q_inv)< 1e-10) = 0;
% Q_inv = sparse(Q_inv);
% disp(check(Q_inv));
% disp(abs(rho) * (abs(rho) + 2));