function sigPulse = MatchedFilter(Rxsig,H)

%%
[l,~] = size(Rxsig);

%%
nsamp = length(H);
if l>1
    Rxsig = fft(Rxsig,nsamp,2);
    H = repmat(H,l,1);
    sigPulse = ifft(Rxsig.*H,[],2);
else
    Rxsig = fft(Rxsig,nsamp);
    sigPulse = ifft(Rxsig.*H);
end

%%

end

%%