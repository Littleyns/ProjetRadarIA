%% Radar Free Space Channel
function y = RadarChannel(x,fc,fs,R,TwoWay,v)

% x,fc,fs,tgtpos,TwoWay,tgtvel
%%
if nargin < 5
    TwoWay = 1;
    v = 0;
elseif nargin < 6
    v = 0;
end

%%
Ts = 1/fs;  
c = physconst('lightspeed');        % Vitesse de la lumière (m/s)
lambda = c/fc;                      % longueur d'onde du radar (m)

%%
if size(R,2) == 2
    xd = R(:,1); yd = R(:,2);
    R = sqrt(xd.^2 + yd.^2);
    an = atan(yd/xd);
    an = cos(an);
else
    R = R(:,1);
    an = 1;
end

%%
if TwoWay == 1
    t = 2*R/c;                          % Retard de propagation aller-retour (s)
    fsloss = (lambda.^2./(4*pi*R).^2);     % trajet aller
    fsloss = fsloss.^2;                       % trajet retour
else
    t = R/c;                          % Retard de propagation aller-retour (s)
    fsloss = (lambda.^2./(4*pi*R).^2);     % trajet aller
end



%%
Ndelay = t/Ts;
N = round(Ndelay);
fracDelay = Ndelay-N;

% VariableFractionalDelay

%%
w = sqrt(fsloss).*exp(-1i*2*pi*2*R/lambda);  % Amplitude et phase dû à la propagation

%%
y = zeros(length(N),length(x));

%%
if all(N < length(x))
    for i = 1:length(N)
        if length(N) == size(x,1)
            y(i,:) = [zeros(1,N(i)) w(i)*x(i,1:end-N(i))];
        elseif size(x,1) == 1
            y(i,:) = [zeros(1,N(i)) w(i)*x(1,1:end-N(i))];
        else
            error('error !')
        end
    end
    if nargin == 6
        v = v*an;
        if TwoWay == 1
            fd = 2*v/lambda;
        else
            fd = v/lambda;
        end
        mfd = fd*(0:length(y)-1);
        Av = exp(-1i*2*pi*(mfd*Ts));
        y = Av.*y;
    end
else
    error(['error:  Rmax = ' num2str(round(length(x)*Ts*c/2*1e-3,2)) ' Km'])
end

%%

% fd = 2*v/lambda;
% m = 0:length(x)-1;
% Av = exp(1i*2*pi*fd*m*Ts);
end

