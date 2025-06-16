original = @(x) x.^2;
[f2,a5] = fourierBuilder(2);
[f5,a10] = fourierBuilder(5);
[f10,a20] = fourierBuilder(10);
figure;
fplot(original,[-1,1]);

hold on;
fplot(f2,[-1,1]);
fplot(f5,[-1,1]);
fplot(f10,[-1,1]);
legend('x^2','N=2','N=5','N=10');
title('Fourier Series Representation of X^2');
xlabel('x');
ylabel('y');

function [fourier,an] = fourierBuilder(n)    
    fourier = @(x) 1/3;
    an = zeros(n);
    for i =1:n
        an(i) = (4*((-1)^i))/((i*pi)^2);
        fourier = @(x) fourier(x) + an(i).*cos(i.*pi.*x);
    end
end