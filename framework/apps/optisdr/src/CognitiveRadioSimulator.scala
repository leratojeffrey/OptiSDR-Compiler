import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object CogRadSimulator extends OptiSDRApplicationRunner with CognRadio
trait CognRadio extends OptiSDRApplication
{
	def main() = 
	{
		//
		val Fc1 = 1000.0f;
    val Fc2 = 2000.0f;
    val Fc3 = 3000.0f;
    val Fc4 = 4000.0f;
    val Fc5 = 5000.0f;
    val Fs3 = 12000.0f; 
    //
		val tm3 = 0.0f::>0.00001f<::0.01024f;
		val l = tm3.length*1.0f
		//
		//println(l)
		val f = (-1.0f*l/2.0f)::>1.0f<::(l/2.0f)
    //   
    val x1 = Cos(2.0f*pi*1000.0f*tm3)
    //
		val y1 = modulate(x1,Fc1,Fs3,AMDSB_SC)
		val y2 = modulate(x1,Fc2,Fs3,AMDSB_SC)
		val y3 = modulate(x1,Fc3,Fs3,AMDSB_SC) 
		val y4 = modulate(x1,Fc4,Fs3,AMDSB_SC) 
		val y5 = modulate(x1,Fc5,Fs3,AMDSB_SC)
		//
		var y = y1 + y2 + y3 + y4 + y5;
		//
		//
    val perplot = periodogram(y,f,"CR Signal",0)
		val tc1 = tic2
		var working = true;
		var i = 1
		var j = 1.0f
		while(working)
    {
    	i = i + 1
    	//
    	if(i%10 == 0)
    	{
    		val tout = toc2(tc1)
    		if(floor(tout) == j)
    		{
    			y = y - y3
    			//if(j.toInt == 30)
    			//{
    				
					
    			//	y = y + y3 // Remove - Replaced by ChannelAllocator Algo. above
    			//} 
    			perplot.refresh(y,"CR Signal",0)   			
    			print(".")
    			j = j + 1.0f
    		}
    		
    	}
    	
    }
    //
	}
}