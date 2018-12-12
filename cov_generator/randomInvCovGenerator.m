function [Q_inv] = randomInvCovGenerator(n, p, max_degree)
% n          -> number of nodes
% p          -> probability of presence of an edge
% max_degree -> maxiumum degree of graph
% G is not necessarily connected
    
    % create random graph
    G = full(sprandsym(n, p)~=0);
    [G, ncc] = make_connected(G);
    for i=1:n
       d = 0;
       for j=1:n
          if G(i, j) == 1 && i ~= j
             if d >= max_degree  
                G(i, j) = 0;
                G(j, i) = 0;
             else
                d = d + 1;
             end
          end
       end
    end
    
%     G(logical(eye(n))) = 1;
%     Q_inv = sprandsym(G, [], 1/6, 3);
    
%     tmp = rand(n, n);
%     a = .01;
%     b = 1;
%     tmp = (tmp + a) / (a+b);
    tmp = 1 - 2*rand(n, n);
    Q_inv = (tmp + tmp') / 2;
    Q_inv(G==0) = 0;
    if min(eigs(Q_inv)) < 1
        Q_inv = Q_inv + (1 - min(eigs(Q_inv))) * eye(n);
    end
    
    
    Q = full(inv(Q_inv));
    
    % normalizing diagonals
    A = diag(sqrt(diag(Q)));
    [r, c, v] = find(Q_inv);
    for k=1:length(r)
       i = r(k);
       j = c(k);
       v(k) = A(i, i) * A(j, j) * v(k);
    end
    Q_inv = sparse(r, c, v);
end