function [is_ok] = check(Q_inv)
    Q = inv(full(Q_inv));
    p = size(Q, 1);
    gamma = kron(Q, Q);
    
    [ei, ej, v] = find(Q_inv);
    Q_inv_c = Q_inv;
    Q_inv_c(Q_inv==0) = 1;
    Q_inv_c(Q_inv~=0) = 0;
    [cei, cej, v] = find(Q_inv_c);
    Q_inv = full(Q_inv);
    A = zeros(size(cei, 1), size(ei, 1));
    for i=1:size(cei, 1)
        a = cei(i, 1);
        b = cej(i, 1);
        for j=1:size(ei, 1)
            c = ei(j, 1);
            d = ej(j, 1);
            A(i, j) = gamma((a-1)*p+b, (c-1)*p+d);
        end
    end
    B = zeros(size(ei, 1), size(ei, 1));
    for i=1:size(ei, 1)
        a = ei(i, 1);
        b = ej(i, 1);
        for j=1:size(ei, 1)
            c = ei(j, 1);
            d = ej(j, 1);
            B(i, j) = gamma((a-1)*p+b, (c-1)*p+d);
        end
    end
    if norm(A*pinv(B), inf) < 1
        is_ok = 1;
    else
        is_ok = 0;
    end
end