import uct.rrsg.dsl.optisdr._

object OptiSDRVectorAppRunner extends OptiSDRApplicationRunner with OptiSDRVectorApp

trait OptiSDRVectorApp extends OptiSDRApplication
{

  def main() {
    //val x = DenseVector[Double](100,true)
    //val y = DenseVector[Double](100,true)

    //val z = x + y
    
    //val v = z(0::10)
    //z(0::5).pprint
	//val m = DenseMatrix.rand(10,10)
	//val v1 = DenseVector.rand(10)
	//v1.pprint
	//val v2 = DenseVector.rand(10)
	//v2.pprint
	//val m1 = DenseMatrix.rand(10000,50)
	//val m2 = DenseMatrix.rand(10000,50)
	
	//val matsum = m1.sumRow(1) // Just sum matrix rows
	
	//val mm = sumRows(0, 100){i => m1(i)}

	
	
	//writeMatrix(m1,"testmatrix.mat")

	//val matdotpro = m1*:*m2
	//matdotpro.pprint
	//val z = m * v

	//z.pprint
       // 10000x1 DenseVector
        val v1 = DenseVector.randf(100000)
        // 1000x1000 DenseMatrix
        val m = DenseMatrix.randf(10000,10000)
        
        // perform some simple infix operations
        val v2 = (v1+10)*2.0f-5.0f
        
        // take the pointwise natural log of v2 and sum results
        val logv2Sum = log(v2).sum
        
        // slice elems 1000-2000 of v2 and multiply by m
        val v3 = v2(10000::20000)*m // 1x1000 DenseVector result
         
        // print the first 10 elements to the screen
        v3(0::10).pprint
  }
}