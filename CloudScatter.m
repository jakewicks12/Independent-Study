xC = [2, 1]; % Center of data (mean)
sig = [2, .5]; % Principal axes
theta = pi/3; % Rotate cloud by pi/3
R = [cos(theta) sin(theta); % Rotation matrix
-sin(theta) cos(theta)];
nPoints = 10000; % Create 10,000 points

X = randn(nPoints,2)*diag(sig)*R + ones(nPoints,2)*diag(xC);
scatter(X(:,1),X(:,2),'k.','LineWidth',2)

xbar = mean(X(:,1));% Compute Mean of each collumn
ybar = mean(X(:,2));

Xavg = ones(nPoints,1)*[xbar,ybar]; % Recompute average matrix
B = X - Xavg; 
[U,S,V] = svd(B/sqrt(nPoints), 'econ');
theta = (0:.01:1)*2*pi; % Create vector of points from 0 to 2pi
Xstd = [cos(theta'),sin(theta')]*S*V';
T = S*V';
hold on;
plot(xbar+Xstd(:,1),ybar + Xstd(:,2),'r-', 'LineWidth',1.2)
plot(xbar+2*Xstd(:,1),ybar + 2*Xstd(:,2),'r-','LineWidth',1.2)
plot(xbar+3*Xstd(:,1),ybar + 3*Xstd(:,2),'r-','LineWidth',1.2)