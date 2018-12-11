
n = 4;

G = zeros(n, n);
rc = [0 0 1 -1];
cc = [1 -1 0 0];
for i=1:n
   for j=1:n
       u = (i-1) * n + j;
       for k=1:4
           if 1 <= i+rc(k) && i+rc(k) <= n && 1 <= j+cc(k) && j+cc(k) <= n
               v = (i+rc(k)-1) * n + j+cc(k);
               G(u, v) = 1;
               G(v, u) = 1;
           end
       end
   end
end
Q_inv = G;
Q_inv(logical(eye(n*n))) = 1;
disp(check(Q_inv));
save('4nn.mat', 'Q_inv');
