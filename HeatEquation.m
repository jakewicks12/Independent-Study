u0x = @(x) exp(-(x.^2));
nx = 4;
nt = 1100;
t = 2;
L = 1;

v = 0.01;
tt = linspace(0,t,nt);
figure
hold on
while nx < 10
    x = linspace(-L,L,nx);
    u = heat(nx,nt,t,L,x,v);
    cla;

    [X,T] = meshgrid(x,tt);
    
    %contour(X,T,u,500), shading interp, colorbar, axis equal

    pcolor(x,tt,u), shading interp, colorbar
    xlabel('x');
    ylabel('t');
    drawnow;
    nx = nx +1;
end
hold off

function u = heat(nx,nt,t,L,x,v)
    dx = (2*L)/(nx-1);
    dt = t/nt;
    u = zeros(nt,nx);
    u(1,:) = exp(-(x.^2));
    
    I = eye(nx-2);
    % A = zeros(nx-2,nx-2);
    
    v = dt/(dx^2);
    
    % if v > 0.5
    %      disp('bad v');
    %      disp(v)
    %      return;
    % end
    e = ones(nx-2,1);
    A = spdiags([e -2*e e], -1:1,nx-2,nx-2);
    
    % for i=1:nx-2
    % 
    %     for j=1:nx-2
    %         if j==i+1
    %             A(i,j) = 1;
    %         elseif j==i-1
    %             A(i,j) = 1;
    %         elseif i==j
    %             A(i,i) = -2;
    %         end
    %     end
    % end
    
    for i =2:nt-1
        u(i,2:nx-1) = ((I+(v.*A))*(u(i-1,2:nx-1).')).';
    end
end
