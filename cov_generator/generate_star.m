n = 70;
Q = zeros(n, n);
for i=1:n
    Q(1, i) = 1/4.;
    Q(i, 1) = 1/4.;
    Q(i, i) = 1;
end
for i=2:n
    for j=i+1:n
        Q(i, j) = 1/16.;
        Q(j, i) = 1/16.;
    end
end
Q_inv = inv(Q);
check(Q_inv);
save('star.mat', 'Q_inv');