%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = accpercent(x,y)
	s = 0;
	for i = 1:length(x)
		if (x(i)==y(i))
			s = s + 1;
		endif;
	endfor;
	s = s/length(x);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
