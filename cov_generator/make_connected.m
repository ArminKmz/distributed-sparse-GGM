function [G, num_of_cc] = make_connected(A)
    n = length(A);
    mark = zeros(1, n);
    components = cell(1, 0);
    for u=1:n
        if mark(u) == 0
           [nodes, mark] = dfs(u, A, mark); 
           components{end+1} = nodes;
        end
    end
    G = A;
    num_of_cc = length(components);
    for i=1:num_of_cc-1
        u = components{i}(1);
        v = components{i+1}(1);
        G(u, v) = 1;
        G(v, u) = 1;
    end
end
