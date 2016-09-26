%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 10 % Change this even in OptiSDR
L = 2^N
x1 = linspace(1,L,L);
y1 = xcorr(x1,x1);
%
mfID = fopen('octave.txt','w');
for i = 1:L
    fprintf(mfID,'%.12f\n',real(y1(i)));
endfor;
fclose(mfID);
%
