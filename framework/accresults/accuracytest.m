%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 10 % Change this even in OptiSDR
L = 2^N
x1 = linspace(1,L,L);
y1 = log10(x1);
%y1 = fft(x1);
%
%save octave.txt y1
mfID = fopen('octave.txt','w');
for i = 1:L
    fprintf(mfID,'%.16f\n',y1(i));
endfor;
fclose(mfID);
%
x = load('octave.txt');
%y = load('optisdr.txt');
%z = accpercent(x,y)
%
%y2 = sqrt(mean((y-x).^2)); % RMSE
%y2 = fft(x1);
%
%
