function y = awgnNoise(sig,reqSNR)

sigPower = sum(abs(sig(:)).^2)/numel(sig);

reqSNR = 10^(reqSNR/10);
noisePower = sigPower/reqSNR;


noise = sqrt(noisePower/2)* complex(randn(size(sig)), randn(size(sig)));
y = sig + noise; 

end