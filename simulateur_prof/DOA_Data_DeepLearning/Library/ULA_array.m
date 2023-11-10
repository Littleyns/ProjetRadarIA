function A = ULA_array(N,d,theta,winType)

%% Uniform Linear Array
% N : Nombre d'éléments
% d : Espacement des éléments en longueur d'onde
% theta : Direction du faisceau en degrée

%%
if nargin == 3
    winType = 1;
end

%%
n = 0:N-1;
n = (n-(N-1)/2);

%%
switch winType
   case 1 
       win = ones(N,1);
   case 2
       win = hamming(N) ;
   case 3
       win = hanning(N) ;
   case 4
        win = kaiser(N,pi) ;
   case 5
       win = blackman(N) ;
end

%%
A = win.*exp(1j*2*pi*n'*d*sin(theta*pi/180));

end