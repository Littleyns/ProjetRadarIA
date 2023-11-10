function w = winGen(window,N,varargin)
    
    % Génération de la fenêtre
    n = (0:N-1)';
    switch window
        case 'rect'
            w = ones(N,1);
        case 'blackman'
            w = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1));
        case 'hamming'
            alpha = 0.54;
            beta = 1 - alpha;
            w = alpha - beta*cos(2*pi*n/(N-1));
        case 'hann'
            w = 0.5*(1 - cos(2*pi*n/(N-1)));
        case 'bartlett'
            w = 1 - 2 * abs(n-(N-1)/2)/(N-1);
        case 'riesz'
            n = linspace(-N/2,N/2,N)';
            w = 1-abs(n/(N/2)).^2;
        case 'blackman-harris'
            a0 = 0.35875;
            a1 = 0.48829;
            a2 = 0.14128;
            a3 = 0.01168;
            w = a0 - a1*cos(2*pi*n/(N-1)) + a2*cos(4*pi*n/(N-1)) - a3*cos(6*pi*n/(N-1));
        case 'kaiser'
            beta = varargin{1};
%             if alphaf > 50
%                 beta = 0.1102*(alphaf-8.7);
%             elseif alphaf<= 50 && alphaf >= 21
%                 beta = 0.5842*(alphaf-21)^0.4 + 0.07886*(alphaf-21);
%             else
%                 beta = 0;
%             end
            alpha = beta * sqrt(1 - ((n - (N-1)/2) / ((N-1)/2)).^2);
            w = approx_besseli(0, alpha) ./ approx_besseli(0, beta);
        case 'chebwin'
            A = varargin{1};
            r = 10^(A/20);
            x0 = cosh(1/(N-1)*acosh(r));

            n = 0:N-1;
            k = 1:(N-1)/2;
            x = x0*cos(k*pi/N);
            Tn = chebyPoly(x,N-1);
            s = Tn*cos(2*k'*pi*(n-(N-1)/2)/N);
            w = (1/N)*(r + 2*s);

%             w =zeros(N,1);
%             for n = 0:N-1
%                 s = 0;
%                 for k = 1:(N-1)/2
%                     x = x0*cos(k*pi/N);
%                     s = s + chebyPoly(x,N-1)*cos(2*k*pi*(n-(N-1)/2)/N);
%                 end
%                 w(n+1) = (1/N)*(r + 2*s);
%             end
            w = w(:)/max(w);
    end
end

%% Approximation polynomiale de la fonction de Bessel
function y = approx_besseli(v, x)
    y = zeros(size(x));
    for k = 0:30
        y = y + ((x/2).^k / factorial(k + v)) .^ 2;
    end
end

%% Dolph-Chebyshev
function Tn = chebyPoly(x,n)
    if abs(x)<=1
        Tn = cos(n*acos(x));
    else
        Tn = cosh(n*acosh(x));
    end
end

%%

