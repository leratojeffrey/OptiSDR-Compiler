//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: Profiling Task Graph Data Structure %%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%% Inlcude some C Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Include Cuda run-time and inline Libraries %%%%%%%%%%%%%%%%%
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

//
#include "netradtasks.h"
//

using namespace std;

//

#ifndef __PROFILETG__
#define __PROFILETG__

class ProfileTaskGraph
{
	int	T;
	vector<int>	T_dest;
	vector<int>	C_j;
	vector<float>	T_exec;
	vector<float>	T_comm;
public:
	ProfileTaskGraph(vector<int> &Tdest,vector<int> &Cj)
	{
		T = 0;
		T_dest = Tdest;
		C_j = Cj;
		T_exec.resize(Cj.size());
		T_comm.resize(Cj.size());
	}

	ProfileTaskGraph(int task, vector<int> &Tdest,vector<int> &Cj)
	{
		T = task;
		T_dest = Tdest;
		C_j = Cj;
		T_exec.resize(Cj.size());
		T_comm.resize(Cj.size());
	}
	//ProfileTaskGraph(vector<int>,vector<int>,vector<float>,vector<float>);
	void setTask(int task){T = task;} //
	int getOptimalCETarget();

private:
	float profileTasks(); // Implement these ...
};
#endif


