n = 64;
Q = zeros(n, n);
for i=1:n
    if i <=41
        Q(1, i) = 1/4.;
        Q(i, 1) = 1/4.;
    end
    Q(i, i) = 1;
end
for i=2:min(n, 41)
    for j=i+1:min(n, 41)
        Q(i, j) = 1/16.;
        Q(j, i) = 1/16.;
    end
end
Q_inv = inv(Q);
disp(Q_inv);
save('star.mat', 'Q_inv');