x0 = [400;27;-136;12;3894;2914;-1284;-237];
tol = 1e-6;

f = @(x) x(1).^2 + x(2).^4 + x(3).^2 + x(4).^6 + x(5).^2 + x(6).^2 + x(7).^2 + x(8).^2 - 36;
syms x1 x2 x3 x4 x5 x6 x7 x8;
F = x1^2 + x2^4 + x3^2 + x4^6 + x5^2 + x6^2 + x7^2 + x8^2 - 36;
X = [x1;x2;x3;x4;x5;x6;x7;x8];
gradientF = gradient(F,X);

[sol,fCurrent] = bfgs(x0,gradientF,X,f,tol);

figure;
plot(fCurrent);
title('Evaluation of Gradient Descent');


function [solution,fCurrent] = bfgs(x0,gradientF,X,f,tol) 
    maxguesses = 100;
    xk = zeros(length(x0),maxguesses);
    xk(:,1) = x0;
    err = 1;
    i = 1;
    tau = 0.5;
    c1 = 1e-4;
    fCurrent = zeros(1,maxguesses);
    fCurrent(1) = f(xk(:,1));
    aki =1;

    while (err > tol) && (i<maxguesses)
        ak = aki;
        gxNow = double(subs(gradientF,X,xk(:,i))); %grad at xk
        pk = -gxNow;
        while (f(xk(:,i)+ak*pk) > (f(xk(:,i)) + c1*ak*pk'*gxNow))
            ak = ak*tau;
        end
        
        xk(:,i+1) = xk(:,i) + ak*pk;
        fCurrent(i+1) = f(xk(:,i+1));
        
        err = norm(gxNow);
        i = i+1;
        
    end
    fCurrent = fCurrent(1,1:i);
    solution = xk(:,1:i);
end