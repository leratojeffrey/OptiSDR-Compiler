import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object myFIRApp extends OptiSDRApplicationRunner with myFIR
trait myFIR extends OptiSDRApplication
{ 
  	def main() = {    
    val v = DenseVector.randf(4096) // Generate 1000 random data
		//val rcoeffs = FIRFilter.constcoeffs()

		val fcoeffs = readcoeffs("dsls/optisdr/kaiser/kaiser244.mat")
		//fcoeffs.pprint
		
		val lpcoeffs = kaisercoeffs(128,fcoeffs,0.4f,0.6f,LowPass)
		//lpcoeffs.pprint
		
		val hpcoeffs = kaisercoeffs(128,fcoeffs,0.4f,0.6f,HighPass)
		//hpcoeffs.pprint	

		val htcoeffs = kaisercoeffs(128,fcoeffs,Hilbert) // 
		//htcoeffs.pprint

		val htcs = readcoeffs("dsls/optisdr/kaiser/kaiser244.mat") // Using HT M and B of Oppenheimer DSP Book
		//htcs.pprint
		val htcfs = kaisercoeffs(128,readcoeffs("dsls/optisdr/kaiser/kaiser244.mat"),Hilbert) // 
		//htcfs.pprint
		//val kcoeffs = kaizer(0.4*constPI,0.6*constPI,0.001,LowPass)
		//kcoeffs.pprint
		//val testI0=I0(1)
		//println(testI0)
		val firout  = DenseVector[Float](4096,true)
		
		FIR(v,htcfs,128,firout) // Filter With Hilbert Transform Coeffs design with Kaiser
		//firout.pprint // Print output data
		//v.pprint
		writeVector(v,"randdata.mat")
		writeVector(firout,"filtereddata.mat")
		//%%%%%%%%%%%%%%%%%%%%%%%

		val outp  = DenseVector[Float](10000,true)
		val inp = SineWave(1,500,0.01f, 0, 0.2f)
		
		FIR(inp,htcfs,128,outp) // Filter With Hilbert Transform Coeffs design with Kaiser
		outp.pprint // Print output data
		//v.pprint
		writeVector(inp(0::1000),"swdata.mat")
		writeVector(outp(0::1000),"swfiltereddata.mat")
		
  	}
}