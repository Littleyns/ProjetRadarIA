function [Pmusic,EN] = Music_doa(Rxx,theta,M)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               MUSIC Doa                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [EN,Pmusic]=Music_doa(Rxx,theta,M)
% est un algorithme qui permet de calculer les angles d'arrivées des
% signaux RF non-corrélées en presence d'un bruit additif gaussien, blanc
% et indépendant du signal. L'algorithme est basé sur la noyion de sous-espace
%------------------- Entrées ------------------------------------%
% Rxx: est la matrice d'auto-corrélation du réseau entier
% theta = -90:0.1:90;
% M: est le Nombre des signaux
%-------------------- sortie ------------------------------------%
% EN: est le sous-espace bruit
% Pmusic: est le Psuedo-spectre MUSIC
%=================================================================

%%----- Décomposition en vecteurs singulières --------%%

%%----- Décomposition en vecteurs et valeurs propres--------%%

[V,Dia] = eig(Rxx); % V: vecteurs propre et Dia: valeurs propre de Rxx

%Trie des valeurs propres par ordre croissant(du plus petit au plus grand)
[~,Index] = sort(diag(Dia));

%Valeurs propres de Rxx
% valP = diag(Dia);

%Les plus petites valeurs propres de Rxx
% sig2 = valP(Index(1:length(Rxx)-M));
% bruit=abs(sig2);
% bruit = min(sig2);

%Calcul de la matrice des N-M vecteurs propres associés au sous-espace bruit
EN = V(:,Index(1:length(Rxx)-M)); % Calcul du sous-espace bruit, EN

%%----- Calcul Psuedo-spectre MUSIC -----%%  

Pmusic = zeros(1,length(theta));
n = (1:length(Rxx))';
for k = 1:length(theta)   
    a = exp(1i*(n-1)*pi*sin(theta(k)*pi/180));
    Pmusic(k) = 1/abs((a'*EN)*(EN'*a)); %Psuedo-spectre MUSIC
%     Pmusic(k) = (a'*a)/abs((a'*EN)*(EN'*a)); %Psuedo-spectre MUSIC
end

