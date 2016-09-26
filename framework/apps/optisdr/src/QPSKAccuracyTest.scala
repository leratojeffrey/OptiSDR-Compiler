import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object QPSKAccuracyT extends OptiSDRApplicationRunner with qpskacctest
trait qpskacctest extends OptiSDRApplication
{
	//
	def rmse(x: Rep[DenseVector[Double]], y: Rep[DenseVector[Double]]): Rep[Double] =
	{
				
		var s = 0.0
		var i = 0
		while(i < x.length)
		{
			s= s + pow(x(i)-y(i),2) //
			i = i + 1
		}
		//
		sqrt(s/x.length)
	}
  	def main() = 
  	{
		val N = 3 // Change this even in Octave
  		val L = pow(2,N).toInt
		val x = randb(L)//onesi(L)//linspace(1,L,L)
		//
		val y = QAM(x,pi/4.0f,1) // QPSK
		//
		val rl = y.real
		x(1::L).pprint
		rl(0::1).pprint
		writeVector(rl,"accresults/optisdr.txt")
		val z1 = readVector("accresults/optisdr.txt") // OptiSDR res
		val z2 = readVector("accresults/octave.txt") // Octave res
		//
		//println("Length :: "+y.length+", RMSE :: "+rmse(z1,z2))
			//
  	}
}
