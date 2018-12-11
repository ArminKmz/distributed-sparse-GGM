function [nodes, mark] = dfs(u, A, mark)
    if mark(u) == 1
        nodes = [];
    else
        tmp = [u];
        mark(u) = 1;
        for v=1:length(A(u, :))
            if (A(u, v) == 1) && (mark(v) == 0)
                [cnodes, mark] = dfs(v, A, mark);
                tmp = [tmp cnodes];
            end
        end
        nodes = tmp;
    end
end