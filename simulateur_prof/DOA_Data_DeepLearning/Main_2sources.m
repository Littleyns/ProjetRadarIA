clc;
clear;

%%
chem = [pwd '\Library'];
addpath(chem);

%% Paramètres du système
fc = 2.725e9;       % fréquence centrale (Hz)
BW = 10e6;          % Largeur de bande (Hz)

Te = 1.4*1e-3;      % Durée de modulation (s) (Pulse width)
T = 4.1*1e-3;       % Période de répétition des impulsions (s)

fs = 4*BW;          % Fréquence d'échantillonnage (Hz)
Ts = 1/fs; 

c = physconst('lightspeed');    % Vitesse de la lumière (m/s)
lambda = c/fc;                  % Longueur d'onde du radar (m)


%% Paramètres réseau d'antenne
N = 10; % Nombre d'éléments
d = 0.5;

%%
SNR = 20;
M = 2;
% R = 10;
% doa = [20 18];
R = [10; 20];

%% Linear FM Pulse Waveform generation
Param.system.LFMDir = 'Up';
LFMDir = Param.system.LFMDir;
LFM = LFMWaveform(BW,Te,fs,LFMDir,T);

txSig = LFM.signal;

%% Radar Matched Filter
[mfcoeff,H] = RadarGetMatchedFilter(LFM.signal);

%% Radar Free Space Channel
txSig_chan = RadarChannel(txSig,fc,fs,R,true);

sigPower = sum(abs(txSig_chan).^2,2)/size(txSig_chan,2);
sigPower = sqrt(sigPower(1)./sigPower(2:end));
txSig_chan(2:end,:) = sigPower.*txSig_chan(2:end,:);

%%
Dtheta = [2 4 6 8 10 12 14 16 18 20 22 24];
theta = -90:1:90;

for j = 1:length(Dtheta)
    theta1 = -60:1:60 - Dtheta(j);
    theta2 = theta1 + Dtheta(j);
    for i = 1:length(theta1)
        doa1 = theta1(i);
        doa2 = theta2(i);      
    
        %%
        doa = [doa1 doa2];
    
        %% Receive array
        sig_Rx = ReceiveArray(txSig_chan,N,d,doa);
        
        %% ADD Noise
        sigNoise = awgnNoise(sig_Rx,SNR);
        
        %% Compression
        X = MatchedFilter(sigNoise,H);
        
        %% Correlation Matrix
        Rxx = (1/length(X))*(X*X');
        data(:,i) = [real(Rxx(:)); imag(Rxx(:))];
        label(:,i) = double(theta == doa1)' + double(theta == doa2)';
    end
%     writematrix(data,['DataDoa2_' num2str(Dtheta(j)) '_' num2str(SNR) '.csv']);
%     writematrix(label,['LabelDoa2_' num2str(Dtheta(j)) '_' num2str(SNR) '.csv']);

end

%%
thetaM = -90:0.1:90;
Pmusic = Music_doa(Rxx,thetaM,M);


%%
plot(thetaM,10*log10(Pmusic));
grid on
title('DOAs cibles')
xlabel('DOA (degree)')
ylabel('|P(\theta)|')






%%
function [y,xx] = ReceiveArray(x,N,d,theta)
%% Receive array
    xx = x;
    
    %%
    A = ULA_array(N,d,theta);
    
    Na = size(A,2);
    [lx,cx] = size(xx);
    
    if Na == lx
        y  = (A*xx);
    elseif Na == cx
        y  = (A*xx.');
    else
        error('Error.');
    end

end

%%
function LFM = LFMWaveform(BW,Te,Fs,ChirpDir,T,NbreTirs)

%% St = A*exp(1i*2*pi*(f0*t+(K/2)*t.^2));
% BW : Largeur de bande (Hz)
% Te : durée de l'impulsion émise (s) 
% fs : fréquence d'échantillonnage (Hz)
% ChirpDir : Type de modulation
% PRI : période de répétition des impulsions

if nargin == 5
 NbreTirs = 1;   
end

%%
f0 = 0;
Ts = 1/Fs;
npoints = Te/Ts;
K = BW/Te;
A = 1;

%% Chirp Waveform : St = A*exp(1i*2*pi*(f0*t+(K/2)*t.^2));
switch ChirpDir
    case 'Symmetry'
        t = linspace(-0.5*Te, 0.5*Te, npoints);
    case 'Up'
        t = linspace(0, Te, npoints);
    case 'Down'
        t = linspace(-Te, 0, npoints);
end

%
LFM.sig = A*exp(1i*2*pi*(f0*t + (K/2)*t.^2));
LFM.temps = linspace(0, Te, npoints);

%%
Npointsrx = fix((T-Te)/Ts);
signal = [LFM.sig zeros(1,Npointsrx)];

LFM.signal = repmat(signal,1,NbreTirs);

LFM.signal2 = [zeros(1,Npointsrx) LFM.sig];


end

%%
function [mfcoeff,H, win] = RadarGetMatchedFilter(waveform,winType)

%%
if nargin == 1
    winType = 1;
end

%%
nsamp = length(waveform);
x = nonzeros(waveform);

n = size(x,1);
if n > 1
    x = x.';
end

%%
N = length(x);
switch winType
   case 1 
       win = winGen('rect',N);
   case 2
       win = winGen('hamming',N);
   case 3
       win = winGen('chebwin',N,60);
   case 4
       win = winGen('kaiser',N,pi);
   case 5
       win = winGen('blackman',N);
end
win = win.';

%%
mfcoeff = conj(fliplr(x));
mfcoeff = win.*mfcoeff;

%% Attention padding
% nfft = 2^nextpow2(nsamp);
tmp = [zeros(1,nsamp-length(x)) mfcoeff];
H = fft(tmp,nsamp);

%%

end
