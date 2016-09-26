//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Module: OptiSDR CUDA Kernels Simplified for  %%%%%%%%%%%%%
//%%%%%%%%%%%%%% Author: Lerato J. Mohapi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%% Institute: University of Cape Town %%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%% Inlcude some C Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "qcustomplot.h"
#include "OptiSDRCuda.h"
#include "optisdrdevices.h"
#include "netradread.h"
#include "radardsp.h"

namespace Ui
{
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT
  
public:
	//
  	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();
	//
	//
	void setupDemo(int demoIndex);
	//void setPlotData(float *_data, int _dlen){data=_data; dlen=_dlen;}
	void InitOptiSDRImageSC(QCustomPlot *customPlot,int nx, int ny);
	void updteWithHilbertTransform(QCustomPlot *customPlot,RadarDSP *processor,NetRADRead *dreader,OptiSDRDevices *devices);
	//
	void setupOptiSDRImageSC(QCustomPlot *customPlot,cuFloatComplex *indata,int nx, int ny);
	void setupOptiSDRImageSC(QCustomPlot *customPlot,float *indata,int dlen);
	void setupRealtimeDataDemo(QCustomPlot *customPlot);
	void setupColorMapDemo(QCustomPlot *customPlot);
	//  
	void setupPlayground(QCustomPlot *customPlot);
	//
private slots:
	void realtimeDataSlot();
	void bracketDataSlot();
	void screenShot();
	void allScreenShots();
	//
private:
	Ui::MainWindow *ui;
	QString demoName;
	QTimer dataTimer;
	QCPItemTracer *itemDemoPhaseTracer;
	int currentDemoIndex;
	QCPColorMap *colorMap2;  
	QCPColorScale *colorScale2;  
  	QCPMarginGroup *marginGroup2;
	//
	void readBinFile(string strFilename, unsigned int numSamples, vector<short> &vsSamples); // This Func
};
#endif // MAINWINDOW_H
