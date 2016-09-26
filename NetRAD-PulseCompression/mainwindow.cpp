//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: OptiSDR CUDA Kernels Simplified for  NetRAD Pulse Compression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%% SSHFS: sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /home/optisdr/OptiSDR/201x_NetRad %
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%% Inlcude some C Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QDesktopWidget>
#include <QScreen>
#include <QMessageBox>
#include <QMetaEnum>
#include <qtconcurrentrun.h>
#include <omp.h>
#include "OptiSDRCuda.h"

void launchGraphData(MainWindow * par, QCustomPlot *customPlot,RadarDSP *processor,NetRADRead *dreader);
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	setGeometry(400, 250, 542, 390);
	//
	struct timeval t1, t2,t3,t4;
	//
	// Begining to Time Code ...
	//
	gettimeofday(&t1, 0);
	gettimeofday(&t3, 0);
	printf("\n[SDR_DSL_INFO]$ Reading Data Into Memory: ...\n");
	NetRADRead *dreader = new NetRADRead(130000,5,1300,2048); // Reader Settings
	RadarDSP *processor = new RadarDSP(dreader);
	OptiSDRDevices *devices = new OptiSDRDevices();
	//
	printf("Data Len is: %d .\n",dreader->dataLen);
	//
	// Read Text File to Vector ....
	readBinFile("/home/optisdr/OptiSDR/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",dreader->dataLen,dreader->vsNetRadSamples);
	//
	//readNetRADTextFile("/home/optisdr/OptiSDR/NetRAD_processing_code/data3/e11_06_04_1724_14_P1_1_130000_S0_1_2047_node3.txt",dreader->vsNetRadSamples,dreader->dataLen);
	//
	gettimeofday(&t4, 0);
	double time1 = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000000.0;
	printf("[SDR_DSL_INFO]$ Overall Reading Data Into Memory = %f s .\n", time1);
	printf("\n[SDR_DSL_INFO]$ Processing: ...\n");
	//setupOptiSDRImageSC(ui->customPlot,getComplexSamples(dreader->vsNetRadSamples,0,dreader->dataLen),dreader->ftpoint,dreader->dsize*dreader->subchunk*dreader->NUM_SIGS);
	//updteWithHilbertTransform(ui->customPlot,getComplexSamples(dreader->vsNetRadSamples,0,dreader->dataLen),dreader->ftpoint,dreader->dsize*dreader->subchunk*dreader->NUM_SIGS);
	//	
	InitOptiSDRImageSC(ui->customPlot,dreader->ftpoint-30,dreader->dsize*dreader->subchunk*dreader->NUM_SIGS);	
	updteWithHilbertTransform(ui->customPlot,processor,dreader,devices);
	//QFuture<void> thr = QtConcurrent::run(launchGraphData, this, ui->customPlot, processor, dreader);
	//setupDemo(0);
  	//
	//writeFileF("data/zeropad.txt",processor->hout1,dreader->ftpoint);
	setWindowTitle("OptiSDR: "+demoName);
	statusBar()->clearMessage();
	currentDemoIndex = 0;
	ui->customPlot->replot();
	//
	// End of Timing
	//);
	//
	printf("\n[SDR_DSL_INFO]$ Plotting: ...\n");
	//
	gettimeofday(&t2, 0);	
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("[SDR_DSL_INFO]$ Overall Processing and Plotting Time = %f s .\n", time);
	//printf("\n[OPTISDR_INFO]$ Number of Processors: %d .\n",processor->numCPU);
	printf("[SDR_DSL_INFO]$ Screenshoting...\n\n");
        QTimer::singleShot(10000, this, SLOT(screenShot()));
	//screenShot();
	printf("[SDR_DSL_INFO]$ Done Screenshoting!\n\n");
}

void launchGraphData(MainWindow * par, QCustomPlot *customPlot,RadarDSP *processor,NetRADRead *dreader)
{
	//par->updteWithHilbertTransform(customPlot,processor->hout1,dreader);
}
//
void MainWindow::readBinFile(string strFilename, unsigned int numSamples, vector<short> &vsSamples)
{
	//Read File into Complex Data
	//
	vsSamples.resize(numSamples);	
	int dataSize=numSamples*sizeof(short);
	short *buffer = (short*)malloc(dataSize);
	//
	struct timeval t3,t4;
	gettimeofday(&t3, 0);
	FILE *fileIO;
	fileIO=fopen(strFilename.c_str(),"r");
	if(!fileIO)
	{
		printf("[SDR_DSL_ERROR]$ Unable to open file!");
		exit(1);
	}	
	fread(buffer,dataSize,1,fileIO); // Serial File I/O: Read data into buffer
	fclose(fileIO);
	//
	gettimeofday(&t4, 0);
	double time1 = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000000.0;
	printf("[SDR_DSL_INFO]$ Overall Reading Data Into Memory = %f s .\n", time1);
	//
	#pragma omp parallel for
	for(int i=0; i<numSamples; i++)
	{
		vsSamples[i] = buffer[i];
	}
	//
	free(buffer);
	//
}
//
void MainWindow::setupDemo(int demoIndex)
{
  switch (demoIndex)
  {
    case 0: setupRealtimeDataDemo(ui->customPlot); break;
    case 1: setupColorMapDemo(ui->customPlot); break;
  }
  setWindowTitle("OptiSDR: "+demoName);
  statusBar()->clearMessage();
  currentDemoIndex = demoIndex;
  ui->customPlot->replot();
}
//
void MainWindow::InitOptiSDRImageSC(QCustomPlot *customPlot,int numx, int ny)
{
	demoName = "Slow/Fast time Plotter";
  int nx = numx;
  printf("1. Testing ny:[%d].\n",ny);
  // configure axis rect:
  customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  customPlot->axisRect()->setupFullAxesBox(true);
  customPlot->xAxis->setLabel("Monostatic Range [m]");
  customPlot->yAxis->setLabel("Pulse Number");
  customPlot->yAxis->setRangeReversed (true);

  // set up the QCPColorMap:
  colorMap2 = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
  customPlot->addPlottable(colorMap2);
  //int nx = 2048;
  //int ny = 130000;
  colorMap2->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
  colorMap2->data()->setRange(QCPRange(0,nx), QCPRange(ny,0)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
  //
  // add a color scale:
  colorScale2 = new QCPColorScale(customPlot);
  customPlot->plotLayout()->addElement(0, 1, colorScale2); // add it to the right of the main axis rect
  colorScale2->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap2->setColorScale(colorScale2); // associate the color map with the color scale
  colorScale2->axis()->setLabel("Color Scale");
  //
  //  
  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  marginGroup2 = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup2);
  colorScale2->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup2);
  
  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();
}
//
//void MainWindow::updteWithHilbertTransform(QCustomPlot *customPlot,cuFloatComplex* indata,int nx, int ny)
void MainWindow::updteWithHilbertTransform(QCustomPlot *customPlot,RadarDSP *processor,NetRADRead *dreader,OptiSDRDevices *devices)
{
  processor->startProcess(dreader,devices,colorMap2); 
  // set the color gradient of the color map to one of the presets: This is the Color Map Type, we use one similar to Matlab
  colorMap2->setGradient(QCPColorGradient::gpJet);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.
  //
  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
  colorMap2->rescaleDataRange();
  //
  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  //QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup2);
  colorScale2->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup2);
  //
  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();

}
//
void MainWindow::setupOptiSDRImageSC(QCustomPlot *customPlot,cuFloatComplex *indata,int nx, int ny)
{
  demoName = "Slow/Fast time Plotter";
  
  // configure axis rect:
  customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  customPlot->axisRect()->setupFullAxesBox(true);
  customPlot->xAxis->setLabel("Fast Time");
  customPlot->yAxis->setLabel("Slow Time");

  // set up the QCPColorMap:
  QCPColorMap *colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
  customPlot->addPlottable(colorMap);
  //int nx = 2048;
  //int ny = 130000;
  colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
  colorMap->data()->setRange(QCPRange(0,nx), QCPRange(ny,0)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
  // now we assign some data, by accessing the QCPColorMapData instance of the color map:
  double x, y, z;
  for (int xIndex=0; xIndex<nx; ++xIndex)
  {
    for (int yIndex=0; yIndex<ny; ++yIndex)
    {
      colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
      //double r = 3*qSqrt(x*x+y*y)+1e-2;
      //z = 2*x*(qCos(r+2)/r-qSin(r+2)/r); // the B field strength of dipole radiation (modulo physical constants)
      z = cuCabsf(indata[xIndex*nx + yIndex]);
      colorMap->data()->setCell(xIndex, yIndex, z);
    }
  }
  
  // add a color scale:
  QCPColorScale *colorScale = new QCPColorScale(customPlot);
  customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
  colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap->setColorScale(colorScale); // associate the color map with the color scale
  colorScale->axis()->setLabel("Color Scale");
  
  // set the color gradient of the color map to one of the presets:
  colorMap->setGradient(QCPColorGradient::gpJet);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.
  
  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
  colorMap->rescaleDataRange();
  
  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  
  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();
}
//
void MainWindow::setupOptiSDRImageSC(QCustomPlot *customPlot,float *indata,int dlen)
{
  demoName = "Slow/Fast time Plotter";
  
  // configure axis rect:
  customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  customPlot->axisRect()->setupFullAxesBox(true);
  customPlot->xAxis->setLabel("Fast Time");
  customPlot->yAxis->setLabel("Slow Time");

  // set up the QCPColorMap:
  QCPColorMap *colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
  customPlot->addPlottable(colorMap);
  int nx = 2048;
  int ny = 8192;
  colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
  colorMap->data()->setRange(QCPRange(0,2048), QCPRange(8192,0)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
  // now we assign some data, by accessing the QCPColorMapData instance of the color map:
  double x, y, z;
  for (int xIndex=0; xIndex<nx; ++xIndex)
  {
    for (int yIndex=0; yIndex<ny; ++yIndex)
    {
      colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
      //double r = 3*qSqrt(x*x+y*y)+1e-2;
      //z = 2*x*(qCos(r+2)/r-qSin(r+2)/r); // the B field strength of dipole radiation (modulo physical constants)
      z = indata[xIndex*nx + yIndex];
      colorMap->data()->setCell(xIndex, yIndex, z);
    }
  }
  
  // add a color scale:
  QCPColorScale *colorScale = new QCPColorScale(customPlot);
  customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
  colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap->setColorScale(colorScale); // associate the color map with the color scale
  colorScale->axis()->setLabel("Color Scale");
  
  // set the color gradient of the color map to one of the presets:
  colorMap->setGradient(QCPColorGradient::gpJet);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.
  
  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
  colorMap->rescaleDataRange();
  
  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  
  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();
}
//
void MainWindow::setupRealtimeDataDemo(QCustomPlot *customPlot)
{
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  QMessageBox::critical(this, "", "You're using Qt < 4.7, the realtime data demo needs functions that are available with Qt 4.7 to work properly");
#endif
  demoName = "Real Time Data Plotter";
  
  // include this section to fully disable antialiasing for higher performance:
  //
  customPlot->addGraph(); // blue line
  customPlot->graph(0)->setPen(QPen(Qt::blue));
  customPlot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot->graph(0)->setAntialiasedFill(false);
  customPlot->addGraph(); // red line
  customPlot->graph(1)->setPen(QPen(Qt::red));
  customPlot->graph(0)->setChannelFillGraph(customPlot->graph(1));
  
  customPlot->addGraph(); // blue dot
  customPlot->graph(2)->setPen(QPen(Qt::blue));
  customPlot->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  customPlot->addGraph(); // red dot
  customPlot->graph(3)->setPen(QPen(Qt::red));
  customPlot->graph(3)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(3)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot->xAxis->setAutoTickStep(false);
  customPlot->xAxis->setTickStep(2);
  customPlot->axisRect()->setupFullAxesBox();
  
  // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->yAxis2, SLOT(setRange(QCPRange)));
  
  // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
  connect(&dataTimer, SIGNAL(timeout()), this, SLOT(realtimeDataSlot()));
  dataTimer.start(0); // Interval 0 means to refresh as fast as possible
}
//
void MainWindow::setupColorMapDemo(QCustomPlot *customPlot)
{
  demoName = "Slow/Fast time Plotter";
  
  // configure axis rect:
  customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  customPlot->axisRect()->setupFullAxesBox(true);
  customPlot->xAxis->setLabel("Fast Time");
  customPlot->yAxis->setLabel("Slow Time");

  // set up the QCPColorMap:
  QCPColorMap *colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
  customPlot->addPlottable(colorMap);
  int nx = 2048;
  int ny = 8192;
  colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
  colorMap->data()->setRange(QCPRange(0,2048), QCPRange(8192,0)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
  // now we assign some data, by accessing the QCPColorMapData instance of the color map:
  double x, y, z;
  for (int xIndex=0; xIndex<nx; ++xIndex)
  {
    for (int yIndex=0; yIndex<ny; ++yIndex)
    {
      colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
      double r = 3*qSqrt(x*x+y*y)+1e-2;
      z = 2*x*(qCos(r+2)/r-qSin(r+2)/r); // the B field strength of dipole radiation (modulo physical constants)
      colorMap->data()->setCell(xIndex, yIndex, z);
    }
  }
  
  // add a color scale:
  QCPColorScale *colorScale = new QCPColorScale(customPlot);
  customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
  colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap->setColorScale(colorScale); // associate the color map with the color scale
  colorScale->axis()->setLabel("Colors");
  
  // set the color gradient of the color map to one of the presets:
  colorMap->setGradient(QCPColorGradient::gpPolar);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.
  
  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
  colorMap->rescaleDataRange();
  
  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  
  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();
}
//
void MainWindow::realtimeDataSlot()
{
  // calculate two new data points:
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  double key = 0;
#else
  double key = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif
  static double lastPointKey = 0;
  if (key-lastPointKey > 0.01) // at most add point every 10 ms
  {
    double value0 = qSin(key); //qSin(key*1.6+qCos(key*1.7)*2)*10 + qSin(key*1.2+0.56)*20 + 26;
    double value1 = qCos(key); //qSin(key*1.3+qCos(key*1.2)*1.2)*7 + qSin(key*0.9+0.26)*24 + 26;
    // add data to lines:
    ui->customPlot->graph(0)->addData(key, value0);
    ui->customPlot->graph(1)->addData(key, value1);
    // set data of dots:
    ui->customPlot->graph(2)->clearData();
    ui->customPlot->graph(2)->addData(key, value0);
    ui->customPlot->graph(3)->clearData();
    ui->customPlot->graph(3)->addData(key, value1);
    // remove data of lines that's outside visible range:
    ui->customPlot->graph(0)->removeDataBefore(key-8);
    ui->customPlot->graph(1)->removeDataBefore(key-8);
    // rescale value (vertical) axis to fit the current data:
    ui->customPlot->graph(0)->rescaleValueAxis();
    ui->customPlot->graph(1)->rescaleValueAxis(true);
    lastPointKey = key;
  }
  // make key axis range scroll with the data (at a constant range size of 8):
  ui->customPlot->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
  ui->customPlot->replot();
  
  // calculate frames per second:
  static double lastFpsKey;
  static int frameCount;
  ++frameCount;
  if (key-lastFpsKey > 2) // average fps over 2 seconds
  {
    ui->statusBar->showMessage(
          QString("%1 FPS, Total Data points: %2")
          .arg(frameCount/(key-lastFpsKey), 0, 'f', 0)
          .arg(ui->customPlot->graph(0)->data()->count()+ui->customPlot->graph(1)->data()->count())
          , 0);
    lastFpsKey = key;
    frameCount = 0;
  }
}

void MainWindow::bracketDataSlot()
{
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  double secs = 0;
#else
  double secs = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif
  
  // update data to make phase move:
  int n = 500;
  double phase = secs*5;
  double k = 3;
  QVector<double> x(n), y(n);
  for (int i=0; i<n; ++i)
  {
    x[i] = i/(double)(n-1)*34 - 17;
    y[i] = qExp(-x[i]*x[i]/20.0)*qSin(k*x[i]+phase);
  }
  ui->customPlot->graph()->setData(x, y);
  
  itemDemoPhaseTracer->setGraphKey((8*M_PI+fmod(M_PI*1.5-phase, 6*M_PI))/k);
  
  ui->customPlot->replot();
  
  // calculate frames per second:
  double key = secs;
  static double lastFpsKey;
  static int frameCount;
  ++frameCount;
  if (key-lastFpsKey > 2) // average fps over 2 seconds
  {
    ui->statusBar->showMessage(
          QString("%1 FPS, Total Data points: %2")
          .arg(frameCount/(key-lastFpsKey), 0, 'f', 0)
          .arg(ui->customPlot->graph(0)->data()->count())
          , 0);
    lastFpsKey = key;
    frameCount = 0;
  }
}
//
void MainWindow::setupPlayground(QCustomPlot *customPlot)
{
  Q_UNUSED(customPlot)
}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::screenShot()
{
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  QPixmap pm = QPixmap::grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#else
  QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#endif
  QString fileName = demoName.toLower()+".png";
  fileName.replace(" ", "");
  fileName.replace("/", "");
  //printf("[SDR_DSL_INFO]$ Screenshot Name: %s.\n\n",fileName.toStdString().c_str());
  pm.save("screenshots/"+fileName);
  //printf("[SDR_DSL_INFO]$ Done Testing...\n\n");
  qApp->quit();
}

void MainWindow::allScreenShots()
{
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  QPixmap pm = QPixmap::grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#else
  QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#endif
  QString fileName = demoName.toLower()+".png";
  fileName.replace(" ", "");
  pm.save("screenshots/"+fileName);
  
  if (currentDemoIndex < 19)
  {
    if (dataTimer.isActive())
      dataTimer.stop();
    dataTimer.disconnect();
    delete ui->customPlot;
    ui->customPlot = new QCustomPlot(ui->centralWidget);
    ui->verticalLayout->addWidget(ui->customPlot);
    setupDemo(currentDemoIndex+1);
    // setup delay for demos that need time to develop proper look:
    int delay = 250;
    if (currentDemoIndex == 10) // Next is Realtime data demo
      delay = 12000;
    else if (currentDemoIndex == 15) // Next is Item demo
      delay = 5000;
    QTimer::singleShot(delay, this, SLOT(allScreenShots()));
  } else
  {
    qApp->quit();
  }
}
