%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 10 % Change this even in OptiSDR
L = 2^N
x1 = randint(L)%ones(L); %linspace(1,L,L);
y1 = pskmod(x1,4,0,"gray");
%
mfID = fopen('octave.txt','w');
for i = 1:L
    fprintf(mfID,'%.12f\n',y1(i));
endfor;
fclose(mfID);
%
