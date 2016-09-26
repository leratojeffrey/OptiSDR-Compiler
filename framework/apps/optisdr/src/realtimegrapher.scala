import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object RTGraphRunner extends OptiSDRApplicationRunner with RTGraph
trait RTGraph extends OptiSDRApplication
{
	def main() = 
	{
		val tc1 = tic2
		//val perplot = periodogram(x1,f,"Sinusoid Signal")
		var working = true;
		var i = 1
		var j = 1.0f
		//var t =  DenseVector[Float](2048,true)
		
    val t = 0.0f::>1.0f<::101.0f
    val l = t.length*1.0f
		val f = (-1.0f*l/2.0f)::>1.0f<::(l/2.0f)
    val preamb0 = 10.0f*Sin((pi/50.0f)*t)
    val preamb1 = 15.0f*Sin((pi/50.0f)*t)
		var preamble = preamb0<<preamb1
		
    /*val perplot = periodogram(preamble.toFloat,t,"Preamble Signal")
		//
    while(working)
    {
    	i = i + 1
    	//val tout = toc2(tc1)
    	//println(i%1001)
    	if(i%10 == 0)
    	{
    		val tout = toc2(tc1)
    		if(floor(tout) == j)
    		{
    			print(".")
					preamble = preamble<<preamb0;
			 		preamble = preamble<<preamb1;
			 		perplot.refresh(preamble,"Preamble Signal")
    			//print(floor(tout))
    			//print(",")
    			//print(j)
    			//print("  ")
    			j = j + 1.0f
    		}
    		//perplot.refresh(,"Sinusoid Signal")
    		//
    	}
    }*/
	}
}