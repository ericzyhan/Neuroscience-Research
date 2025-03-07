function nonlog = g(a, p, q, N)
    k = log(N);
    
    nonlog = (N^(1/2 + N) * (k)^(-(1/2) - a*N) * (N - k)^(-(1/2) - N + k) * ...
    (1/2 * p * (1 - q) + p * q)^(1/2 * k * (-1 + k)) * ...
    (1 - (1/2 * p * (1 - q) + p * q)^(k))^(N - k)) / sqrt(2 * pi);
end

% Define parameters a and N
a = 0.2;  % Example value for a
N = 100;   % Example value for N

% Define the range for p and q (e.g., 0 to 1 with 100 points)
p_values = linspace(0, 1, 100);
q_values = linspace(0, 1, 100);

% Initialize a matrix to store results
g_values = zeros(length(p_values), length(q_values));

% Calculate g(a, p, q, N) for each combination of p and q
for i = 1:length(p_values)
    for j = 1:length(q_values)
        g_values(i, j) = g(a, p_values(i), q_values(j), N);
    end
end

% Plot the surface
figure;
surf(q_values, p_values, g_values);
xlabel('q');
ylabel('p');
zlabel('g(a, p, q, N)');
title('Surface plot of g(a, p, q, N) vs. p and q');
