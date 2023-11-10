clc;
clear;
cls

%%
figure(1)
grid on
title('DOAs cibles')
xlabel('DOA (degree)')
ylabel('|P(\theta)|')

%%
M = 1;
thetaM = -90:0.1:90;

%%
data = readmatrix('Data_Doa.csv');
label = readmatrix('Label_Doa.csv');

%%
c = size(M,2);

for i = 1:size(data,2)
    temp = data(:,i);
    Rxx = reshape(temp(1:100),10,[])+1i*reshape(temp(100+1:end),10,[]);

    lab = 60*label(:,i);

    Pmusic = Music_doa(Rxx,thetaM,M);
    plot(thetaM,10*log10(Pmusic));

    grid
    drawnow
    pause(0.1) 

end



%%
Dtheta = [2 4 6 8 10 12 14 16 18 20 22 24];
datatemp = readmatrix('DataDoa1.csv');
labeltemp = readmatrix('LabelDoa1.csv');

datatemp = [datatemp readmatrix('DataDoa1_10.csv')];
labeltemp = [labeltemp readmatrix('LabelDoa1_10.csv')];


for i = 1:length(Dtheta)

    datatemp = [datatemp readmatrix(['DataDoa2_' num2str(Dtheta(i)) '.csv'])];
    labeltemp =[labeltemp readmatrix(['LabelDoa2_' num2str(Dtheta(i)) '.csv'])];
end


for i = 1:length(Dtheta)

    datatemp = [datatemp readmatrix(['DataDoa2_' num2str(Dtheta(i)) '_20.csv'])];
    labeltemp =[labeltemp readmatrix(['LabelDoa2_' num2str(Dtheta(i)) '_20.csv'])];
end

n = randperm(size(datatemp,2));

data = datatemp(:,n);
label = labeltemp(:,n);

writematrix(data,'Data\Data_Doa.csv');
writematrix(label,'Data\Label_Doa.csv');




%%

