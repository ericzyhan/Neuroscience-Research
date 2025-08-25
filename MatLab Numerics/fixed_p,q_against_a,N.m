% Define the function f(a, p, q, N)
function logtrue = f(a, p, q, N)
    logtrue = log(sqrt(2*pi*N) + N*log(N/exp(1))) - ...
           (log(sqrt(2*pi*a*N)) + N*a*log((N*a)/exp(1))) + ...
           log(sqrt(2*pi*N*(1-a))) + N*(1-a)*log((N*(1-a))/exp(1)) + ...
           (((a*N)*(a*N-1))/2)*log(p*q) + N*(1-a)*log(1-(p*((1-q)/2)+p*q)^(a*N));
end

function nonlog = g(a, p, q, N)
    % Define the constants u, v, w, bin_exp, and x
    u = sqrt(2*pi*N) * (N/exp(1))^N;
    v = sqrt(2*pi*a*N) * ((a*N)/exp(1))^(a*N) * sqrt(2*pi*(N*(1-a))) * ((N*(1-a))/exp(1))^(N*(1-a));
    w = p * q;
    bin_exp = ((a*N)^2 - a*N) / 2;
    x = (1 - (p*((1 - q)/2) + p*q)^(a*N))^(N*(1 - a));
    
    % Compute the result
    nonlog = (u / v) * (w^bin_exp) * x;
end

% Set p and q values
p = 1;
q = 0.9;

% Create the meshgrid for a and N
a_vals = linspace(0, 1, 10);  % a values from 0 to 1
N_vals = 2:60;  % N values from 2 to 100

[A, N] = meshgrid(a_vals, N_vals);
Z = arrayfun(@(a, n) f(a, p, q, n), A, N);  % Compute f(a, p, q, N) for each grid point

% Plot the surface
figure
surf(A, N, Z);
colormap('parula');  % Use a built-in colormap like 'parula'

% Labels and title
xlabel('a');
ylabel('N');
zlabel('f(a, N)');
title('3D Surface Plot of f(a, N)');
