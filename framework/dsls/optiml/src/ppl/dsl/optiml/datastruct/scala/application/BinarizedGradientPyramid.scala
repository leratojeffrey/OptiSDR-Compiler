package ppl.dsl.optiml.datastruct.scala.application

import ppl.dsl.optiml.datastruct.scala._
import ppl.dsl.optila.datastruct.scala._

class BinarizedGradientPyramid (
  val pyramid: DenseVector[GrayscaleImage],
  val start_level: Int,
  val levels: Int,
  val fixedLevelIndex: Int
)
 
