#ifndef __NETRADREAD__
#define __NETRADREAD__
#include <iostream>

class NetRADRead
{
public:
	int dsize, chunk, subchunk, ftpoint, dataLen, NUM_SIGS;
	NetRADRead(int len, int _dsize,int numsigs, int ftp)
	{
		dsize = _dsize;
		NUM_SIGS = numsigs;
		subchunk = len/NUM_SIGS/dsize;
		ftpoint=ftp;
		chunk=NUM_SIGS*ftpoint;		
		dataLen=subchunk*dsize*chunk;
	}
	//
	vector<short> vsNetRadSamples;
//private:
	
};
#endif
