import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object NedRADProcessor extends OptiSDRApplicationRunner with nedrad
trait nedrad extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		// Read raw NedRAD data
  		val idata =  load("/srv/rrsg/data/projects_general/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1610_29_P1_1_1300_S0_1_2047_node1.bin")
  		//Hilbert Transform the data
			val htout  = khilbert(idata(0::2048))
			//Take DFT of Hilbert Transformed data samples
			//val dft1 = dft(htout)
			//Read Chirp Reference Signal
			val chirpref =  load("e11_06_04_1610_29_P1_1_1300_S0_1_2047_node1.bin")
			//Take DFT of Chirp Reference Signal
			val dft2 = dft(chirpref(0::2048))
			// Mix the DFTs (dft1 and dft2)
			
			//Take iDFT of the Mixed DFTs
			
			//Plot Input
			//plot(inp)
			
			// Plot ht of input
			//plot(htout.im)
			
			//plot dft1
			//plot(dft1.rl)
			
			//plot dft 2
			//plot(dft2.rl)
			
			//
  	}
}