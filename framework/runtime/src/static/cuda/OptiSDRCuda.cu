#ifndef _OPTISDR_CUDA_
#define _OPTISDR_CUDA_
//
//
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
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%% Include Delite Tools %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
#include "cudahelperFuncs.h"
#include "cudaDeliteArrayfloat.h"
#include "cppDeliteArrayfloat.h"
//
#include "OptiSDRCuda.h"
//
using namespace std;
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%% OptiSDR USRP1 Functions Implementations %%%%%%%%%%%
//
// TODO: This is where comments started.
namespace po = boost::program_options;
struct optisdr_usrp_config configs;
static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}
//
// Read Data In Parallel Into DeliteArray
//
void readData(cuFloatComplex *outdata, std::vector<std::complex<short> > &buffer, unsigned int from, unsigned int to)
{
	int j = 0;	
	#pragma omp parallel for
	for(unsigned int i=from; i<to; i++)
	{
		//outdata[i] = buffer[j];
		outdata[i].x = (float)buffer[j].real();
		outdata[i].y = (float)buffer[j].imag();
		//printf("%f ",outdata[i].x);//
		j++;
	}
}
//
void recv_to_file(uhd::usrp::multi_usrp::sptr usrp, std::string &cpu_format, std::string &wire_format,cuFloatComplex *outdata, size_t samps_per_buff, unsigned long long num_requested_samples, double time_requested, bool bw_summary, bool stats,
    bool null, bool enable_size_map, bool continue_on_bad_packet)
{
    // TODO: This must have been initialized when passed as arguments
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // TODO: This must have been initialized when passed as arguments
    //
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    unsigned long long num_total_samps = 0;
    unsigned int from = 0;
    //create a receive streamer
    uhd::stream_args_t stream_args(cpu_format,wire_format);
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);
		//
    uhd::rx_metadata_t md;
    std::vector<std::complex<short> > buff(samps_per_buff);
    //
    bool overflow_message = true;

    //setup streaming
    uhd::stream_cmd_t stream_cmd((num_requested_samples == 0)?
        uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS:
        uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE
    );
    stream_cmd.num_samps = num_requested_samples;
    stream_cmd.stream_now = true;
    stream_cmd.time_spec = uhd::time_spec_t();
    rx_stream->issue_stream_cmd(stream_cmd);

    boost::system_time start = boost::get_system_time();
    unsigned long long ticks_requested = (long)(time_requested * (double)boost::posix_time::time_duration::ticks_per_second());
    boost::posix_time::time_duration ticks_diff;
    boost::system_time last_update = start;
    unsigned long long last_update_samps = 0;

    typedef std::map<size_t,size_t> SizeMap;
    SizeMap mapSizes;
    while(not stop_signal_called and (num_requested_samples != num_total_samps or num_requested_samples == 0)) {
        boost::system_time now = boost::get_system_time();

        size_t num_rx_samps = rx_stream->recv(&buff.front(), buff.size(), md, 3.0, enable_size_map); //
	//
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            std::cout << boost::format("Timeout while streaming") << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW){
            if (overflow_message) {
                overflow_message = false;
                std::cerr << boost::format(
                    "Got an overflow indication. Please consider the following:\n"
                    "  Your write medium must sustain a rate of %fMB/s.\n"
                    "  Dropped samples will not be written to the file.\n"
                    "  Please modify this example for your purposes.\n"
                    "  This message will not appear again.\n"
                ) % (usrp->get_rx_rate()*sizeof(std::complex<short>)/1e6);
            }
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
            std::string error = str(boost::format("Receiver error: %s") % md.strerror());
            if (continue_on_bad_packet){
                std::cerr << error << std::endl;
                continue;
            }
            else
                throw std::runtime_error(error);
        }

        if (enable_size_map) {
            SizeMap::iterator it = mapSizes.find(num_rx_samps);
            if (it == mapSizes.end())
                mapSizes[num_rx_samps] = 0;
            mapSizes[num_rx_samps] += 1;
        }
			// Read Samples Into Array: TODO - We need this for OptiSDR
    	//printf("Do I get here...\n");
			readData(outdata,buff,from,from+(unsigned int)num_rx_samps);
      num_total_samps += num_rx_samps;
			from+=(unsigned int)num_rx_samps;
			//printf("from: %i \n",from);

        if (bw_summary) {
            last_update_samps += num_rx_samps;
            boost::posix_time::time_duration update_diff = now - last_update;
            if (update_diff.ticks() > boost::posix_time::time_duration::ticks_per_second()) {
                double t = (double)update_diff.ticks() / (double)boost::posix_time::time_duration::ticks_per_second();
                double r = (double)last_update_samps / t;
                std::cout << boost::format("\t%f Msps") % (r/1e6) << std::endl;
                last_update_samps = 0;
                last_update = now;
            }
        }

        ticks_diff = now - start;
        if (ticks_requested > 0){
            if ((unsigned long long)ticks_diff.ticks() > ticks_requested)
                break;
        }
    }
		// TODO: Move this to the a new Function called End
    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);
    //
    if (stats) {
        std::cout << std::endl;

        double t = (double)ticks_diff.ticks() / (double)boost::posix_time::time_duration::ticks_per_second();
        std::cout << boost::format("Received %d samples in %f seconds") % num_total_samps % t << std::endl;
        double r = (double)num_total_samps / t;
        std::cout << boost::format("%f Msps") % (r/1e6) << std::endl;

        if (enable_size_map) {
            std::cout << std::endl;
            std::cout << "Packet size map (bytes: count)" << std::endl;
            for (SizeMap::iterator it = mapSizes.begin(); it != mapSizes.end(); it++)
                std::cout << it->first << ":\t" << it->second << std::endl;
        }
    }
}
//
typedef boost::function<uhd::sensor_value_t (const std::string&)> get_sensor_fn_t;
//
bool check_locked_sensor(std::vector<std::string> sensor_names, const char* sensor_name, get_sensor_fn_t get_sensor_fn, double setup_time)
{
    if (std::find(sensor_names.begin(), sensor_names.end(), sensor_name) == sensor_names.end())
        return false;

    boost::system_time start = boost::get_system_time();
    boost::system_time first_lock_time;

    std::cout << boost::format("Waiting for \"%s\": ") % sensor_name;
    std::cout.flush();

    while (true) {
        if ((not first_lock_time.is_not_a_date_time()) and
                (boost::get_system_time() > (first_lock_time + boost::posix_time::seconds(setup_time))))
        {
            std::cout << " locked." << std::endl;
            break;
        }
        if (get_sensor_fn(sensor_name).to_bool()){
            if (first_lock_time.is_not_a_date_time())
                first_lock_time = boost::get_system_time();
            std::cout << "+";
            std::cout.flush();
        }
        else {
            first_lock_time = boost::system_time();	//reset to 'not a date time'

            if (boost::get_system_time() > (start + boost::posix_time::seconds(setup_time))){
                std::cout << std::endl;
                throw std::runtime_error(str(boost::format("timed out waiting for consecutive locks on sensor \"%s\"") % sensor_name));
            }
            std::cout << "_";
            std::cout.flush();
        }
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }
    std::cout << std::endl;
    return true;
}
//
//
//
double initUSRP(int argc, char *argv[])
{
	struct timeval kernt1,kernt2;
	gettimeofday(&kernt1,0);
	//
    //
     uhd::set_thread_priority_safe();
    //variables to be set by po
    std::string args, ant, subdev, ref;
    std::string type;
    //size_t total_num_samps, spb;
    double rate, freq, gain, bw, setup_time;
    //
    //
    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", po::value<std::string>(&configs.file)->default_value("usrp_samples.dat"), "name of the file to write binary samples to")
        ("type", po::value<std::string>(&configs.type)->default_value("short"), "sample type: double, float, or short")
        ("nsamps", po::value<size_t>(&configs.total_num_samps)->default_value(64*10000), "total number of samples to receive")
        ("duration", po::value<double>(&configs.total_time)->default_value(0), "total number of seconds to receive")
        ("time", po::value<double>(&configs.total_time), "(DEPRECATED) will go away soon! Use --duration instead")
        ("spb", po::value<size_t>(&configs.spb)->default_value(10000), "samples per buffer")
        ("rate", po::value<double>(&rate)->default_value(8e6), "rate of incoming samples")
        ("freq", po::value<double>(&freq)->default_value(2600e6), "RF center frequency in Hz")
        ("gain", po::value<double>(&gain)->default_value(45), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "daughterboard antenna selection")
        ("subdev", po::value<std::string>(&subdev), "daughterboard subdevice specification")
        ("bw", po::value<double>(&bw), "daughterboard IF filter bandwidth in Hz")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", po::value<std::string>(&configs.wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
        ("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("progress", "periodically display short-term bandwidth")
        ("stats", "show average bandwidth on exit")
        ("sizemap", "track packet size and display breakdown on exit")
        ("null", "run without writing to file")
        ("continue", "don't abort on a bad packet")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")) {
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cout
            << std::endl
            << "This application streams data from a single channel of a USRP device to a file.\n"
            << std::endl;
        return ~0;
    }

    configs.bw_summary = vm.count("progress") > 0;
    configs.stats = vm.count("stats") > 0;
    configs.null = vm.count("null") > 0;
    configs.enable_size_map = vm.count("sizemap") > 0;
    configs.continue_on_bad_packet = vm.count("continue") > 0;

    if (configs.enable_size_map)
        std::cout << "Packet size tracking enabled - will only recv one packet at a time!" << std::endl;

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    configs.usrp = uhd::usrp::multi_usrp::make(args);

    //Lock mboard clocks
    configs.usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) configs.usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % configs.usrp->get_pp_string() << std::endl;

    //set the sample rate
    if (rate <= 0.0){
        std::cerr << "Please specify a valid sample rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    configs.usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (configs.usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the center frequency
    if (vm.count("freq")) { //with default of 0.0 this will always be true
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        configs.usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (configs.usrp->get_rx_freq()/1e6) << std::endl << std::endl;
    }

    //set the rf gain
    if (vm.count("gain")) {
        std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        configs.usrp->set_rx_gain(gain);
        std::cout << boost::format("Actual RX Gain: %f dB...") % configs.usrp->get_rx_gain() << std::endl << std::endl;
    }

    //set the IF filter bandwidth
    if (vm.count("bw")) {
        std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % bw << std::endl;
        configs.usrp->set_rx_bandwidth(bw);
        std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % configs.usrp->get_rx_bandwidth() << std::endl << std::endl;
    }

    //set the antenna
    if (vm.count("ant")) configs.usrp->set_rx_antenna(ant);

    boost::this_thread::sleep(boost::posix_time::seconds(setup_time)); //allow for some setup time

    //check Ref and LO Lock detect
    if (not vm.count("skip-lo")){
        check_locked_sensor(configs.usrp->get_rx_sensor_names(0), "lo_locked", boost::bind(&uhd::usrp::multi_usrp::get_rx_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "mimo")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "mimo_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "external")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "ref_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
    }
    	//
	//
	configs.outdata = (cuFloatComplex*)malloc(configs.total_num_samps*sizeof(cuFloatComplex));
	configs.format = "sc16";
	//
	gettimeofday(&kernt2,0);
	//
	double sttime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec)/1e6;
	printf("[OPTISDR-INFOR]$ Done Setting Up the the USRP in %fs.\n\n\n[OPTISDR-INFOR]$  Running ...\n\n",sttime);
	return sttime;
}
/*
double initUSRP(int argc, char *argv[])
{
	struct timeval kernt1,kernt2;
	gettimeofday(&kernt1,0);
	//
    //
     uhd::set_thread_priority_safe();
    //variables to be set by po
    std::string args, ant, subdev, ref;
    std::string type;
    //size_t total_num_samps, spb;
    double rate, freq, gain, bw, setup_time;
    //
    //
    //setup the program options
    options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", value<std::string>(&configs.file)->default_value("usrp_samples.dat"), "name of the file to write binary samples to")
        ("type", value<std::string>(&configs.type)->default_value("short"), "sample type: double, float, or short")
        ("nsamps", value<size_t>(&configs.total_num_samps)->default_value(64*10000), "total number of samples to receive")
        ("duration", value<double>(&configs.total_time)->default_value(0), "total number of seconds to receive")
        ("time", value<double>(&configs.total_time), "(DEPRECATED) will go away soon! Use --duration instead")
        ("spb", value<size_t>(&configs.spb)->default_value(10000), "samples per buffer")
        ("rate", value<double>(&rate)->default_value(8e6), "rate of incoming samples")
        ("freq", value<double>(&freq)->default_value(2600e6), "RF center frequency in Hz")
        ("gain", value<double>(&gain)->default_value(45), "gain for the RF chain")
        ("ant", value<std::string>(&ant), "daughterboard antenna selection")
        ("subdev", value<std::string>(&subdev), "daughterboard subdevice specification")
        ("bw", value<double>(&bw), "daughterboard IF filter bandwidth in Hz")
        ("ref", value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", value<std::string>(&configs.wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
        ("setup", value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("progress", "periodically display short-term bandwidth")
        ("stats", "show average bandwidth on exit")
        ("sizemap", "track packet size and display breakdown on exit")
        ("null", "run without writing to file")
        ("continue", "don't abort on a bad packet")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
    ;
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    //print the help message
    if (vm.count("help")) {
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cout
            << std::endl
            << "This application streams data from a single channel of a USRP device to a file.\n"
            << std::endl;
        return ~0;
    }

    configs.bw_summary = vm.count("progress") > 0;
    configs.stats = vm.count("stats") > 0;
    configs.null = vm.count("null") > 0;
    configs.enable_size_map = vm.count("sizemap") > 0;
    configs.continue_on_bad_packet = vm.count("continue") > 0;

    if (configs.enable_size_map)
        std::cout << "Packet size tracking enabled - will only recv one packet at a time!" << std::endl;

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    configs.usrp = uhd::usrp::multi_usrp::make(args);

    //Lock mboard clocks
    configs.usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) configs.usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % configs.usrp->get_pp_string() << std::endl;

    //set the sample rate
    if (rate <= 0.0){
        std::cerr << "Please specify a valid sample rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    configs.usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (configs.usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the center frequency
    if (vm.count("freq")) { //with default of 0.0 this will always be true
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        configs.usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (configs.usrp->get_rx_freq()/1e6) << std::endl << std::endl;
    }

    //set the rf gain
    if (vm.count("gain")) {
        std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        configs.usrp->set_rx_gain(gain);
        std::cout << boost::format("Actual RX Gain: %f dB...") % configs.usrp->get_rx_gain() << std::endl << std::endl;
    }

    //set the IF filter bandwidth
    if (vm.count("bw")) {
        std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % bw << std::endl;
        configs.usrp->set_rx_bandwidth(bw);
        std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % configs.usrp->get_rx_bandwidth() << std::endl << std::endl;
    }

    //set the antenna
    if (vm.count("ant")) configs.usrp->set_rx_antenna(ant);

    boost::this_thread::sleep(boost::posix_time::seconds(setup_time)); //allow for some setup time

    //check Ref and LO Lock detect
    if (not vm.count("skip-lo")){
        check_locked_sensor(configs.usrp->get_rx_sensor_names(0), "lo_locked", boost::bind(&uhd::usrp::multi_usrp::get_rx_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "mimo")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "mimo_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "external")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "ref_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
    }
    	//
	//
	configs.outdata = (cuFloatComplex*)malloc(configs.total_num_samps*sizeof(cuFloatComplex));
	configs.format = "sc16";
	//
	gettimeofday(&kernt2,0);
	//
	double sttime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec)/1e6;
	printf("[OPTISDR-INFOR]$ Done Setting Up the the USRP in %fs.\n\n\n[OPTISDR-INFOR]$  Running ...\n\n",sttime);
	return sttime;
}*/
//
//
double streamin_usrp()
{
	struct timeval kernt1,kernt2;
	gettimeofday(&kernt1,0);
	//
	//
	// TODO: This goes to streamin_uspr(...) func in OptiSDRCuda.cu files
	if (configs.total_num_samps == 0)
	{
        	std::signal(SIGINT, &sig_int_handler);
        	std::cout << "Press Ctrl + C to stop streaming..." << std::endl;
    	}
	//std::cout<<"[OPTISDR-INFO]$ Testing If I get Here."<<std::endl;
	#define recv_to_file_args(format) \
		(configs.usrp,format, configs.wirefmt, configs.file, configs.spb, configs.total_num_samps, configs.total_time, configs.bw_summary, configs.stats, configs.null, configs.enable_size_map, configs.continue_on_bad_packet)
	//recv to file
	//printf(" Data Type: %s.",configs.type.c_str());
	//if (configs.type == "double") recv_to_file<std::complex<double> >recv_to_file_args("fc64");
	//else if (configs.type == "float") recv_to_file<std::complex<float> >recv_to_file_args("fc32");
	if (configs.type == "short")
	{
		recv_to_file(configs.usrp,configs.format, configs.wirefmt, configs.outdata, configs.spb, configs.total_num_samps, configs.total_time, configs.bw_summary, configs.stats, configs.null, configs.enable_size_map, configs.continue_on_bad_packet);
	}
	else throw std::runtime_error("Unknown type " + configs.type);
	//
	//finished
	//std::cout << std::endl << "Done!" << std::endl << std::endl;
	//
	//
	gettimeofday(&kernt2,0);
	//
	double sttime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec)/1000000;
	return sttime;
}
void initusrp(int arg)
{
	char *argv[] = {"nothing","--freq=2600000000","--gain=45"};
	printf("\n[OPTISDR-INFOR]$ Setting Up the the USRP :\n");
	initUSRP(3,argv);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Implementing USRP Init Function With System Parameters %%%%%%%%%%%%
//
double startUSRP(int argc, char *argv[], double Fc, double Fs, double Gn,double Bw, int Ns)
{
	struct timeval kernt1,kernt2;
	gettimeofday(&kernt1,0);
	//
    //
     uhd::set_thread_priority_safe();
    //variables to be set by po
    std::string args, ant, subdev, ref;
    std::string type;
    //size_t total_num_samps, spb;
    double rate, freq, gain, bw, setup_time;
    //
    //
    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", po::value<std::string>(&configs.file)->default_value("usrp_samples.dat"), "name of the file to write binary samples to")
        ("type", po::value<std::string>(&configs.type)->default_value("short"), "sample type: double, float, or short")
        ("nsamps", po::value<size_t>(&configs.total_num_samps)->default_value(Ns), "total number of samples to receive")
        ("duration", po::value<double>(&configs.total_time)->default_value(0), "total number of seconds to receive")
        ("time", po::value<double>(&configs.total_time), "(DEPRECATED) will go away soon! Use --duration instead")
        ("spb", po::value<size_t>(&configs.spb)->default_value(10000), "samples per buffer")
        ("rate", po::value<double>(&rate)->default_value(Fs), "rate of incoming samples")
        ("freq", po::value<double>(&freq)->default_value(Fc), "RF center frequency in Hz")
        ("gain", po::value<double>(&gain)->default_value(Gn), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "daughterboard antenna selection")
        ("subdev", po::value<std::string>(&subdev), "daughterboard subdevice specification")
        ("bw", po::value<double>(&bw), "daughterboard IF filter bandwidth in Hz")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", po::value<std::string>(&configs.wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
        ("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("progress", "periodically display short-term bandwidth")
        ("stats", "show average bandwidth on exit")
        ("sizemap", "track packet size and display breakdown on exit")
        ("null", "run without writing to file")
        ("continue", "don't abort on a bad packet")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")) {
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cout
            << std::endl
            << "This application streams data from a single channel of a USRP device to a file.\n"
            << std::endl;
        return ~0;
    }

    configs.bw_summary = vm.count("progress") > 0;
    configs.stats = vm.count("stats") > 0;
    configs.null = vm.count("null") > 0;
    configs.enable_size_map = vm.count("sizemap") > 0;
    configs.continue_on_bad_packet = vm.count("continue") > 0;

    if (configs.enable_size_map)
        std::cout << "Packet size tracking enabled - will only recv one packet at a time!" << std::endl;

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    configs.usrp = uhd::usrp::multi_usrp::make(args);

    //Lock mboard clocks
    configs.usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) configs.usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % configs.usrp->get_pp_string() << std::endl;

    //set the sample rate
    if (rate <= 0.0){
        std::cerr << "Please specify a valid sample rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    configs.usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (configs.usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the center frequency
    if (vm.count("freq")) { //with default of 0.0 this will always be true
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        configs.usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (configs.usrp->get_rx_freq()/1e6) << std::endl << std::endl;
    }

    //set the rf gain
    if (vm.count("gain")) {
        std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        configs.usrp->set_rx_gain(gain);
        std::cout << boost::format("Actual RX Gain: %f dB...") % configs.usrp->get_rx_gain() << std::endl << std::endl;
    }

    //set the IF filter bandwidth
    //if (vm.count("bw")) {
        std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % Bw << std::endl;
        configs.usrp->set_rx_bandwidth(Bw);
        std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % configs.usrp->get_rx_bandwidth() << std::endl << std::endl;
    //}

    //set the antenna
    if (vm.count("ant")) configs.usrp->set_rx_antenna(ant);

    boost::this_thread::sleep(boost::posix_time::seconds(setup_time)); //allow for some setup time

    //check Ref and LO Lock detect
    if (not vm.count("skip-lo")){
        check_locked_sensor(configs.usrp->get_rx_sensor_names(0), "lo_locked", boost::bind(&uhd::usrp::multi_usrp::get_rx_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "mimo")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "mimo_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "external")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "ref_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
    }
    	//
	//
	configs.outdata = (cuFloatComplex*)malloc(configs.total_num_samps*sizeof(cuFloatComplex));
	configs.format = "sc16";
	//
	gettimeofday(&kernt2,0);
	//
	double sttime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec)/1e6;
	printf("[OPTISDR-INFOR]$ Done Setting Up the the USRP in %fs.\n\n\n[OPTISDR-INFOR]$  Running ...\n\n",sttime);
	return sttime;
}
//
/*
double startUSRP(int argc, char *argv[], double Fc, double Fs, double Gn,double Bw, int Ns)
{
	struct timeval kernt1,kernt2;
	gettimeofday(&kernt1,0);
	//
    //
     uhd::set_thread_priority_safe();
    //variables to be set by po
    std::string args, ant, subdev, ref;
    std::string type;
    //size_t total_num_samps, spb;
    double rate, freq, gain, bw, setup_time;
    //
    //
    //setup the program options
    options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", value<std::string>(&configs.file)->default_value("usrp_samples.dat"), "name of the file to write binary samples to")
        ("type", value<std::string>(&configs.type)->default_value("short"), "sample type: double, float, or short")
        ("nsamps", value<size_t>(&configs.total_num_samps)->default_value(Ns), "total number of samples to receive")
        ("duration", value<double>(&configs.total_time)->default_value(0), "total number of seconds to receive")
        ("time", value<double>(&configs.total_time), "(DEPRECATED) will go away soon! Use --duration instead")
        ("spb", value<size_t>(&configs.spb)->default_value(10000), "samples per buffer")
        ("rate", value<double>(&rate)->default_value(Fs), "rate of incoming samples")
        ("freq", value<double>(&freq)->default_value(Fc), "RF center frequency in Hz")
        ("gain", value<double>(&gain)->default_value(Gn), "gain for the RF chain")
        ("ant", value<std::string>(&ant), "daughterboard antenna selection")
        ("subdev", value<std::string>(&subdev), "daughterboard subdevice specification")
        ("bw", value<double>(&bw), "daughterboard IF filter bandwidth in Hz")
        ("ref", value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", value<std::string>(&configs.wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
        ("setup", value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("progress", "periodically display short-term bandwidth")
        ("stats", "show average bandwidth on exit")
        ("sizemap", "track packet size and display breakdown on exit")
        ("null", "run without writing to file")
        ("continue", "don't abort on a bad packet")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
    ;
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    //
    //print the help message
    if(vm.count("help"))
    {
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cout
            << std::endl
            << "This application streams data from a single channel of a USRP device to a file.\n"
            << std::endl;
        return ~0;
    }
    //
    configs.bw_summary = vm.count("progress") > 0;
    configs.stats = vm.count("stats") > 0;
    configs.null = vm.count("null") > 0;
    configs.enable_size_map = vm.count("sizemap") > 0;
    configs.continue_on_bad_packet = vm.count("continue") > 0;
    //
    if (configs.enable_size_map)
        std::cout << "Packet size tracking enabled - will only recv one packet at a time!" << std::endl;
    //
    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    configs.usrp = uhd::usrp::multi_usrp::make(args);
    //
    //Lock mboard clocks
    configs.usrp->set_clock_source(ref);
    //
    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) configs.usrp->set_rx_subdev_spec(subdev);
    //
    std::cout << boost::format("Using Device: %s") % configs.usrp->get_pp_string() << std::endl;
    //
    //set the sample rate
    if (rate <= 0.0){
        std::cerr << "Please specify a valid sample rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    configs.usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (configs.usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the center frequency
    if (vm.count("freq")) { //with default of 0.0 this will always be true
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        configs.usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (configs.usrp->get_rx_freq()/1e6) << std::endl << std::endl;
    }

    //set the rf gain
    if (vm.count("gain")) {
        std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        configs.usrp->set_rx_gain(gain);
        std::cout << boost::format("Actual RX Gain: %f dB...") % configs.usrp->get_rx_gain() << std::endl << std::endl;
    }

    //set the IF filter bandwidth
    //if (vm.count("bw")) {
        std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % Bw << std::endl;
        configs.usrp->set_rx_bandwidth(Bw);
        std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % configs.usrp->get_rx_bandwidth() << std::endl << std::endl;
    //}

    //set the antenna
    if (vm.count("ant")) configs.usrp->set_rx_antenna(ant);

    boost::this_thread::sleep(boost::posix_time::seconds(setup_time)); //allow for some setup time

    //check Ref and LO Lock detect
    if (not vm.count("skip-lo")){
        check_locked_sensor(configs.usrp->get_rx_sensor_names(0), "lo_locked", boost::bind(&uhd::usrp::multi_usrp::get_rx_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "mimo")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "mimo_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
        if (ref == "external")
            check_locked_sensor(configs.usrp->get_mboard_sensor_names(0), "ref_locked", boost::bind(&uhd::usrp::multi_usrp::get_mboard_sensor, configs.usrp, _1, 0), setup_time);
    }
    	//
	//
	configs.outdata = (cuFloatComplex*)malloc(configs.total_num_samps*sizeof(cuFloatComplex));
	configs.format = "sc16";
	//
	gettimeofday(&kernt2,0);
	//
	double sttime = (1000000.0*(kernt2.tv_sec-kernt1.tv_sec) + kernt2.tv_usec-kernt1.tv_usec)/1e6;
	printf("[OPTISDR-INFOR]$ Done Setting Up the the USRP in %fs.\n\n\n[OPTISDR-INFOR]$  Running ...\n\n",sttime);
	return sttime;
}
*/
//
void startusrp(int tst, cppDeliteArraydouble in, cppDeliteArraydouble out)
{
	char *argv[] = {"initusrp"};
	printf("\n[OPTISDR-INFOR]$ Setting Up the the USRP %d %d:\n",out.length,in.length);
	//printf("\nFc=%f\nFs=%f\nGn=%f\nBw=%f\nNs=%f\n\n",in.apply(0),in.apply(1),in.apply(2),in.apply(3),in.apply(4));
  //
	double Fc = in.apply(0), Fs=in.apply(1),Gn=in.apply(2),Bw=in.apply(3),Ns=(int)in.apply(4);
	//printf("\n %f, %f, %f, %f, % \n",Fc,Fs,Gn,Bw,Ns);
	//cout<<Fc<<" "<<Fs<<" "<<Gn<<" "<<Bw<<" "<<Ns<<" "<<endl;
	startUSRP(1,argv,Fc,Fs,Gn,Bw,Ns);
}
// TODO: This is where I uncommented from. 
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%% OptiSDR Suppot Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
StreamProcessorConfig Configs;
//
void initStreamProcessor(int len, int _dsize,int numsigs, int ftp)
{
	Configs.dsize = _dsize;
	Configs.NUM_SIGS = numsigs;
	Configs.subchunk = len/Configs.NUM_SIGS/Configs.dsize;
	Configs.ftpoint=ftp;
	Configs.chunk=Configs.NUM_SIGS*Configs.ftpoint;		
	Configs.dataLen=Configs.subchunk*Configs.dsize*Configs.chunk;
	// Initialize Memory Vectors Here
	Configs.hout1.resize(Configs.subchunk);
	Configs.refsig = getComplexEmpty(2*Configs.dataLen);
	Configs.hdata0.resize(Configs.dsize);
	Configs.hdata1.resize(Configs.dsize);
	Configs.hdata2.resize(Configs.dsize);
}
//
// TODO: 
//
void zpad(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip)
{
	int index = 2*skip;
	int iters = inlen/skip;
	//printf("%d",iters);
	for(int i=0; i<iters; i++)
	{
		for(int j=0; j<skip; j++)
		{
			outp[i*index + j] = inp[i*skip + j];
		}
	}
	//
}
//	
void resize(cuFloatComplex *inp, cuFloatComplex *outp, int inlen, int skip)
{
	int index = 2*skip;
	int iters = inlen/skip;
	//printf("%d",iters);
	for(int i=0; i<iters; i++)
	{
		for(int j=0; j<skip; j++)
		{
			outp[i*index + j] = inp[j];
		}
	}
	//
}
//
//
void readNetRadSamples(string strFilename, unsigned int uiNSamples, vector<short> &vsSamples)
{
	//Read
	ifstream oIFS;
	oIFS.open(strFilename.c_str(), ifstream::binary);
	if(!oIFS.is_open())
	{	
		cout << "[SDR_DSL_INFO]$ Error unable to open file \"" << strFilename << "\"" << endl;
		oIFS.close();
		exit(1);
	}
	//
	vsSamples.resize(uiNSamples);
	//
	oIFS.read((char*)&vsSamples.front(), sizeof(short) * uiNSamples);

	if(oIFS.gcount() << sizeof(short) * uiNSamples && oIFS.eof())
	{
		cout << "[SDR_DSL_INFO]$ Warning: hit end of file after " << oIFS.gcount() / sizeof(short) << " samples. Output is shortened accordingly." << endl;
		vsSamples.resize(oIFS.gcount() / sizeof(short));
	}
	//
	oIFS.close();
}
//
cuFloatComplex* getComplexSamples(vector<short> &vSamples, int _dlen)
{
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	for(int i = 0; i< _dlen; i++)
	{
		cSamples[i] = make_cuFloatComplex(vSamples[i],0.0);
	}
	return cSamples;
}
//
cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to)
{
	int dlen = to-from;
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	for(int i = from; i< to; i++)
	{
		cSamples[i%dlen] = make_cuFloatComplex(vSamples[i],0.0);
	}
	return cSamples;
}
//
//
cuFloatComplex* getComplexEmpty(int _dlen)
{
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	/*for(int i = 0; i< _dlen; i++)
	{
		cSamples[i] = make_cuFloatComplex(0.0,0.0);
	}*/
	return cSamples;
}
//
void readNetRadSamples2(string strFilename, unsigned int nsamples, vector<float> &vsSamples)
{
	//Read
	ifstream oIFS;
	oIFS.open(strFilename.c_str(),ifstream::in);
	if(!oIFS.is_open())
	{	
		cout << "[SDR_DSL_INFO]$ Error unable to open file \"" << strFilename << "\"" << endl;
		oIFS.close();
		exit(1);
	}
	//
	vsSamples.resize(nsamples);
	int i = 0;
	while(i < nsamples)
	{
		oIFS>>vsSamples[i];
		//vsSamples[i] = atof(tmp[i].c_str());
		//printf("%f, ",vsSamples[i]);
		i++;
	}
	oIFS.close();
}
//
cuFloatComplex* getReferenceSignal(int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2("data/rref3sig.dat",_dlen,rSamples);
	readNetRadSamples2("data/iref3sig.dat",_dlen,iSamples);
	//
	for(int i = 0; i< _dlen; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
	}
	return refsig;
}
//
// This one accepts the return size and file size.
//
cuFloatComplex* getReferenceSignal(int fsize, int _dlen)
{
	cuFloatComplex *refsig = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	//
	vector<float> rSamples,iSamples;
	readNetRadSamples2("data/rref3sig.dat",fsize,rSamples);
	readNetRadSamples2("data/iref3sig.dat",fsize,iSamples);
	//
	for(int i = 0; i< fsize; i++)
	{
		refsig[i] = make_cuFloatComplex(rSamples[i],iSamples[i]);
		// 
	}
	//
	for(int j = (_dlen-fsize); j< _dlen; j++)
	{
		refsig[j] = make_cuFloatComplex(0.0f,0.0f);
	}
	//
	return refsig;
}
//
cuFloatComplex* resizeVector(cuFloatComplex *inp, int oldlen, int newlen)
{
	cuFloatComplex *outp = (cuFloatComplex*)malloc(newlen*sizeof(cuFloatComplex));
	//
	for(int i = 0; i< newlen; i++)
	{
		outp[i] = inp[i%oldlen];
	}
	//
	//outp[0] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	//outp[2048] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	//outp[4096] = make_cuFloatComplex(2.0f,2.0f); // Just for testing: TODO: Remove when done
	return outp;
}

//
cuFloatComplex* getComplexSamples(vector<short> &vSamples, int from, int to, int chunksize, int outputsize)
{
	int dlen = outputsize*((to-from)/chunksize);
	int skip = outputsize-chunksize;
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	//for(int i = 0;i<dlen;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	int idx = 0;
	if(skip>0)
	{
		for(int i = from; i< to; i++)
		{
		
			if(((i%chunksize)==0)&&(i>from))
			{
				idx+=skip; // We skip 
				cSamples[idx] = make_cuFloatComplex((float)vSamples[i],0.0f);
				idx+=1;
			}
			else
			{
				cSamples[idx] = make_cuFloatComplex((float)vSamples[i],0.0f);
				idx+=1;
			}
		}
	}
	return cSamples;
}
//
cuFloatComplex* getComplexEmpty(int _dlen, int chunksize, int outputsize)
{
	int dlen = (_dlen)*(outputsize/chunksize);
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(dlen*sizeof(cuFloatComplex));
	//for(int i = 0;i<dlen;i++) cSamples[i] = make_cuFloatComplex(0.0f,0.0f);
	//
	return cSamples;
}
//
cuFloatComplex* getChunk(vector<short> &inp, int from, int to) // Want to get complex values from shorts vector
{
	//
	int chunk = to - from;
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//	
	int index = 0;
	for(int i = from; i< to; i++)
	{
		outp[index] = make_cuFloatComplex((float)inp[i],0.0f);
		index+=1;
	}
	//
	//
	return outp;
}
//
//
cuFloatComplex* getChunk(cuFloatComplex *inp, int from, int to)
{
	//
	int chunk = to-from;
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//
	copy(inp + from,inp + to, outp + 0);
	//
	return outp;
}

//
cuFloatComplex* getChunk(cuFloatComplex *inp, int chunk, int from, int to)
{
	//
	//
	cuFloatComplex *outp = (cuFloatComplex*)malloc(chunk*sizeof(cuFloatComplex));
	//
	//copy(inp + from,inp + to, outp + 0);
	int index = 0;
	for(int i = from; i< to; i++)
	{
		outp[index] = inp[i];
		index+=1;
	}
	//
	return outp;
}
//
cuFloatComplex* getZeroPadded(cuFloatComplex *inp, int initlen, int newlen, int skip)
{
	//
	//
	cuFloatComplex *cSamples = (cuFloatComplex *)malloc(newlen*sizeof(cuFloatComplex));
	//
	int idx = 0;
	if(skip>0)
	{
		for(int i = 0; i< initlen; i++)
		{
		
			if(((i%skip)==0)&&(i>0))
			{
				idx+=skip; // We skip 
				cSamples[idx] = inp[i];
				idx+=1;
			}
			else
			{
				cSamples[idx] = inp[i];
				idx+=1;
			}
		}
	}
	return cSamples;
}
//
void append(cuFloatComplex *A, cuFloatComplex *B, int sizeA, int sizeB, int from)
{
	int index = 0;
	if((sizeA-sizeB)>=sizeB)
	{
		for(int i=from;i<sizeB;i++)
		{
			A[i] = B[index];
			index++;
		}		
		//printf("index = %d\n",index);
	}
}
//
void writeFileF(const char *fpath, float *data,	const unsigned int len)
{
    printf("[SDR_DSL_INFO]$ Output file: %s\n", fpath);
    FILE *fo;

    unsigned int i=0;

    if ( (fo = fopen(fpath, "w")) == NULL) {printf("[SDR_DSL_INFO]$ IO Error\n"); /*return(CUTFalse);*/}

    for (i=0; i<len; ++i)
    {
	if ( (fprintf(fo,"%.7e\n", data[i])) <= 0 )
	{
	    printf("[SDR_DSL_INFO]$ File write Error.\n");
	    fclose(fo);
	    //return(CUTFalse);
	}
    }

    fclose(fo);
    //return(CUTTrue);
}
//
void writeFileF(const char *fpath,cuFloatComplex *xdata, const unsigned int len)
{
    printf("[SDR_DSL_INFO]$ Output file: %s\n", fpath);
    FILE *fo;

    unsigned int i=0;

    if ( (fo = fopen(fpath, "w")) == NULL) {printf("[SDR_DSL_INFO]$ IO Error\n");}

    for (i=0; i<len; ++i)
    {
	//if((fprintf(fo,"%.7e + %.7ei\n",cuCrealf(xdata[i]),cuCimagf(xdata[i]))) <= 0 )
	if((fprintf(fo,"%f \n",cuCabsf(xdata[i]))) <= 0 )
	{
		printf("[SDR_DSL_INFO]$ File write Error.\n");
		fclose(fo);
	}
    }

    fclose(fo);
    //return(CUTTrue);
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% Stream Processor Data Types %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Multiplication Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_multiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		outp[tid] = cuCmulf(_cx[tid],_cy[tid]);
}
//
// This Kernel Utilize a 1-D Grid to a 1-D data indexes (1-D BlockDim = 1024 threads)
// We can multiply 16 Million floating point values in Parallel
__global__ void complexvector_multiply1d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	if(index < N)
	{
		outp[index].x = (_cx[index].x*_cy[index].x) - (_cx[index].y*_cy[index].y);//cuCmulf(_cx[index],_cy[index]);
		outp[index].y = (_cx[index].x*_cy[index].y) + (_cx[index].y*_cy[index].x);
	}
}
// This Kernel Utilize a 2-D Grid flattened to a 1-D data indexes (1-D BlockDim = 1024 threads)
// We can multiply 68.747788288 Million floating point values in Parallel
__global__ void complexvector_multiply2d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	//
	int xsize = blockDim.x*gridDim.x; // X dimension total number of thread
	//
	int index = tidy*xsize + tidx; // Index through 2-D grid
	//
	if(index < N)
		outp[index] = cuCmulf(_cx[index],_cy[index]);
}
// This Kernel Utilize a 3-D Grid flattened to a 1-D data indexes
// We can multiply 129 Million floating point values in Parallel
__global__ void complexvector_multiply3d1d(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	int tidz = threadIdx.z + blockIdx.z * blockDim.z;
	//
	int xsize = blockDim.x*gridDim.x; // X dimension total number of thread
	int zsize = xsize*blockDim.y*gridDim.y; // Entire 2-D grid numer of thread
	//
	int xyindex = tidy*xsize + tidx; // Index through 2-D grid
	int index = tidz*zsize + xyindex; // Index through entire 3-D grid
	//
	if(index < N)
		outp[index] = cuCmulf(_cx[index],_cy[index]);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Conjugate Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_conjugate(cuFloatComplex *_cx, cuFloatComplex *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		//outp[tid].x= (_cx[tid].x*_cy[tid].x)-(_cx[tid].y*_cy[tid].y);
		//outp[tid].y= (_cx[tid].x*_cy[tid].y)+(_cy[tid].x*_cx[tid].y);
		outp[tid] = cuConjf(_cx[tid]);
		tid += blockDim.x * gridDim.x;
	}
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% Complex Absolute Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void complexvector_abs(cuFloatComplex *_cx,float *outp,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		//outp[tid].x= (_cx[tid].x*_cy[tid].x)-(_cx[tid].y*_cy[tid].y);
		//outp[tid].y= (_cx[tid].x*_cy[tid].y)+(_cy[tid].x*_cx[tid].y);
		outp[tid] = cuCabsf(_cx[tid]);
		tid += blockDim.x * gridDim.x;
	}
}
//
__global__ void optisdrifftscale(cuFloatComplex *invec, cuFloatComplex *out, int fp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		out[tid] = make_cuFloatComplex(cuCrealf(invec[tid])/(float)fp,cuCimagf(invec[tid])/(float)fp);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Simplified DSP CUDA Kernel Calls %%%%%%%%%%%%%%%%%%%%%%
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%% FFT Stream Processing using CUDA GPUs %%%%%%%%%%%%%%%%%%%%%%%%
//
void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk)
{
	//
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	//hdata.resize(dsize); // This must be done outside with data initialization/generation
	//hout.resize(dsize); // Might be a good idea to initialize outside of this function...
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);	
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	// Trying to Create Page-Locked std::vector - No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);//cudaHostAlloc(...);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	
	for(int i = 0; i<dsize; i++)
	{
		cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
    		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%% Batched FFT Stream processing using CUDA Streams and Pinned-Memory %%%%%%%%%%%%%%%%%%%%%%%%%%%%
void streamprocessor(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	//struct timeval t1, t2;
	//gettimeofday(&t1, 0);
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	int N_SIGS = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	vector<cuFloatComplex*> ddata,dout;
	ddata.resize(dsize);
	dout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C, N_SIGS);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		//free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//// XCorr Conjugate and Multiply
__global__ void xcorrmultiply(cuFloatComplex *_cx,cuFloatComplex *_cy,cuFloatComplex *outp,int N)
{
	//
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	if(index < N)
	{
		outp[index] = cuCmulf(_cx[index],cuConjf(_cy[index]));
	}
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void XCorrSP(vector<cuFloatComplex*> hdata, cuFloatComplex* refsig, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	//printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//
	cuFloatComplex *drefsig,*dftrefsig;
	cudaMalloc((void**)&drefsig,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dftrefsig,chunk*sizeof(cuFloatComplex));
	//cudaHostRegister(refsig,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	cudaMemcpy(drefsig,refsig,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&drout,chunk*sizeof(cuFloatComplex));
	//
	vector<cuFloatComplex*> ddata,dout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1, ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,chunk/ftpoint);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	
	res = cufftExecC2C(plans[0],drefsig,dftrefsig,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		//complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		xcorrmultiply<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dftrefsig,drout[i],chunk);
		//
		res = cufftExecC2C(plans[i],drout[i],ddata[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Inverse transform fail.\n");}
		//
		// TODO: Try using the Block Size =128, i.e optisdrifftscale<<<chunk/128,128,0,optisdr_treams[i]>>>(...);
        	optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(ddata[i],dout[i],ftpoint,chunk);
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	cudaHostUnregister(refsig);
	cudaFree(drefsig);
	//gettimeofday(&t2, 0);
	//double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	//double time2 = ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))/1000000.0;
	//printf("\n[SDR_DSL_INFO]$ Exec. Time for StreamProcessed FFT = %f s ~= %f...!\n", time,time2);
	//cudaDeviceReset();
}
//
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// Creating a Dummy h vector for computing the Hilbert Transform
//
cuFloatComplex* getHilbertHVector(int _dlen)
{
	//
	cuFloatComplex *h = (cuFloatComplex *)malloc(_dlen*sizeof(cuFloatComplex));
	//
	h[0] = make_cuFloatComplex(1.0f,0.0f);
	h[_dlen/2] = make_cuFloatComplex(1.0f,0.0f);
	//
	int i = 1;
	int j = (_dlen/2)+1;
	//
	while(i<(_dlen/2))
	{
		h[i] = make_cuFloatComplex(2.0f,0.0f);
		i = i + 1;
		h[j] = make_cuFloatComplex(0.0f,0.0f);
		j = j + 1;
	}
	return h;
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void HilbertSP(vector<cuFloatComplex*> hdata, vector<cuFloatComplex*> hout, int dsize,int chunk, int ftpoint)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//int grid1 = ceil(sqrt(numEls/(float)threadsPerBlock)); // For 2-D Grid
	//int grid1 = ceil(cbrt(numEls/(float)threadsPerBlock)); // For 3-D Grid
	dimGrid.x = grid1;
	//dimGrid.y = grid1; // uncomment for 2-D grid
	//dimGrid.z = grid1; // uncomment for 3-D grid
	//
	//blocksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z; // Never really use this
        //TODOMSG("Calculate grid dimensions")
	//printf("\nThe Grid Dim: [%d]\nThreadsPerBlock: [%d]\nBlocksPerGrid: [%d]\n\n",grid1,threadsPerBlock,blocksPerGrid);
	//
	//	
	cuFloatComplex* h = resizeVector(getHilbertHVector(ftpoint),ftpoint,chunk);
	//
	cudaStream_t optisdr_streams[dsize];
	cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*dsize);
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	cuFloatComplex *dh;
	cudaMalloc((void**)&dh,chunk*sizeof(cuFloatComplex));
	cudaHostRegister(h,chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
	//cudaMalloc((void**)&drout,chunk*sizeof(cuFloatComplex));
	//
	vector<cuFloatComplex*> ddata,dout,dhout,drout;
	ddata.resize(dsize);
	dout.resize(dsize);
	dhout.resize(dsize);
	drout.resize(dsize);
	//
	for(int i = 0;i<dsize;i++)
	{
		cudaStreamCreate(&optisdr_streams[i]);
	}
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;	
	for(int i = 0; i<dsize; i++)
	{
		//cufftPlan1d(&plans[i],chunk,CUFFT_C2C,1);
		//cufftPlanMany(&plans[i],1,n,NULL,1,SIG_LEN,NULL,1,SIG_LEN,CUFFT_C2C,N_SIGS);
		res = cufftPlanMany(&plans[i], 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
		//
		cufftSetStream(plans[i],optisdr_streams[i]);
	}
	// Trying to Create Page-Locked std::vector
	// No need for this if we need simple malloc()...
	for(int i = 0;i<dsize;i++)
	{
		// TODO: Check Out cudaMallocHost(...), cudaMallocManage(...), ...
		cudaHostRegister(hdata[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable); // Page-Locked Mem.
		cudaHostRegister(hout[i],chunk*sizeof(cuFloatComplex),cudaHostRegisterPortable);
		//cudaStreamCreate(&optisdr_streams[i]);
		//cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
	}
	//
	cudaMemcpyAsync(dh,h,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[0]);
	for(int i = 0;i<dsize;i++)
	{
		cudaMalloc((void**)&ddata[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&dhout[i],chunk*sizeof(cuFloatComplex));
		cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
		cudaMemcpyAsync(ddata[i],hdata[i],chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice,optisdr_streams[i]);
		//
	}
	//
	// Execution
	for(int i = 0; i<dsize; i++)
	{		
		//
		res = cufftExecC2C(plans[i],ddata[i],dout[i],CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		complexvector_multiply1d1d<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(dout[i],dh,dhout[i],chunk);
		//
		res = cufftExecC2C(plans[i],dhout[i],drout[i],CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
		//
		optisdrifftscale<<<dimGrid,dimBlock,0,optisdr_streams[i]>>>(drout[i],dout[i],ftpoint,chunk);
		//
		cudaMemcpyAsync(hout[i],dout[i],chunk*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost,optisdr_streams[i]);
		cudaStreamSynchronize(optisdr_streams[i]);
	}
	//
	// Releasing Computing Resources
	for(int i = 0; i < dsize; i++)
	{
		cudaStreamDestroy(optisdr_streams[i]);
		cudaHostUnregister(hdata[i]);
		cudaHostUnregister(hout[i]);
		cudaFree(ddata[i]);
		cudaFree(dout[i]);
		cudaFree(dhout[i]);
		cudaFree(drout[i]);
		free(hdata[i]);
		cufftDestroy(plans[i]);
	}
	cudaHostUnregister(h);
	cudaFree(dh);
	free(h);
	//
}
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%% Simplified DSP CUDA Kernel Calls %%%%%%%%%%%%%%%%%%%%%%
__global__ void copyctor(cuFloatComplex *invec,cudaDeliteArrayfloat outvec,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		outvec.update(2*tid,cuCrealf(invec[tid]));
		outvec.update(2*tid+1,cuCimagf(invec[tid]));
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void copyrtoc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cuFloatComplex *h_signal,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		h_signal[tid] = make_cuFloatComplex(x.apply(tid),y.apply(tid));
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void copyrtoc(cudaDeliteArrayfloat x,cuFloatComplex *h_signal,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		h_signal[tid] = make_cuFloatComplex(x.apply(tid),0.0);
		tid += blockDim.x * gridDim.x;
	}
}
//
__global__ void copydelitector(cuFloatComplex *invec,cudaDeliteArrayfloat outrl,cudaDeliteArrayfloat outim,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		outrl.update(tid,cuCrealf(invec[tid]));
		outim.update(tid,cuCimagf(invec[tid]));
		tid += blockDim.x * gridDim.x;
	}
}
//
__global__ void complex2delitearray(cuFloatComplex *invec,cudaDeliteArrayfloat outrl,cudaDeliteArrayfloat outim,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
	{
		outrl.update(tid,invec[tid].x);
		outim.update(tid,invec[tid].y);
	}
}
//
// TODO: This is where another commented section started
void recv_whileloop(uhd::usrp::multi_usrp::sptr usrp, std::string &cpu_format, std::string &wire_format,cuFloatComplex *outdata, size_t samps_per_buff, unsigned long long num_requested_samples, double time_requested, bool bw_summary, bool stats,
    bool null, bool enable_size_map, bool continue_on_bad_packet)
{
    // TODO: This must have been initialized when passed as arguments
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // TODO: This must have been initialized when passed as arguments
    //
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    unsigned long long num_total_samps = 0;
    unsigned int from = 0;
    //create a receive streamer
    uhd::stream_args_t stream_args(cpu_format,wire_format);
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    uhd::rx_metadata_t md;
    std::vector<std::complex<short> > buff(samps_per_buff);
		//
    bool overflow_message = true;
		//
    //setup streaming
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = num_requested_samples;
    stream_cmd.stream_now = true;
    stream_cmd.time_spec = uhd::time_spec_t();
    rx_stream->issue_stream_cmd(stream_cmd);

    boost::system_time start = boost::get_system_time();
    unsigned long long ticks_requested = (long)(time_requested * (double)boost::posix_time::time_duration::ticks_per_second());
    boost::posix_time::time_duration ticks_diff;
    boost::system_time last_update = start;
    unsigned long long last_update_samps = 0;

    typedef std::map<size_t,size_t> SizeMap;
    SizeMap mapSizes;
    while(not stop_signal_called and (num_requested_samples != num_total_samps or num_requested_samples == 0)) {
        boost::system_time now = boost::get_system_time();

        size_t num_rx_samps = rx_stream->recv(&buff.front(), buff.size(), md, 3.0, enable_size_map); //
	//
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            std::cout << boost::format("Timeout while streaming") << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW){
            if (overflow_message) {
                overflow_message = false;
                
            }
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
            std::string error = str(boost::format("Receiver error: %s") % md.strerror());
            if (continue_on_bad_packet){
                std::cerr << error << std::endl;
                continue;
            }
            else
                throw std::runtime_error(error);
        }

        if (enable_size_map) {
            SizeMap::iterator it = mapSizes.find(num_rx_samps);
            if (it == mapSizes.end())
                mapSizes[num_rx_samps] = 0;
            mapSizes[num_rx_samps] += 1;
        }
	// Read Samples Into Array: TODO - We need this for OptiSDR
    	//printf("Do I get here...\n");
	readData(outdata,buff,from,from+(unsigned int)num_rx_samps);
        num_total_samps += num_rx_samps;
	from+=(unsigned int)num_rx_samps;
	//printf("from: %i \n",from);

        if (bw_summary) {
            last_update_samps += num_rx_samps;
            boost::posix_time::time_duration update_diff = now - last_update;
            if (update_diff.ticks() > boost::posix_time::time_duration::ticks_per_second()) {
                double t = (double)update_diff.ticks() / (double)boost::posix_time::time_duration::ticks_per_second();
                double r = (double)last_update_samps / t;
                std::cout << boost::format("\t%f Msps") % (r/1e6) << std::endl;
                last_update_samps = 0;
                last_update = now;
            }
        }

        ticks_diff = now - start;
        if (ticks_requested > 0){
            if ((unsigned long long)ticks_diff.ticks() > ticks_requested)
                break;
        }
    }
}
//

void rx_while()
{
	if (configs.type == "short")
	{
		recv_whileloop(configs.usrp,configs.format, configs.wirefmt, configs.outdata, configs.spb, configs.total_num_samps, configs.total_time, configs.bw_summary, configs.stats, configs.null, configs.enable_size_map, configs.continue_on_bad_packet);
	}
	else throw std::runtime_error("Unknown type " + configs.type);
}
//

void usrpstreamer(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y)
{
	//DeliteCudaTic("usrpstream");
	rx_while();
	//TODO: Copy DataBack to Device
	//
	//
	//printf("X Len = %d ::: Y Len = %d .\n",x.length,y.length);
	dim3 dimBlock, dimGrid;
  int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
  // Set up grid
  // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(configs.total_num_samps/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	cuFloatComplex *inusrp;
	//cudaMalloc((void**)&inusrp,configs.total_num_samps*sizeof(cuFloatComplex));
	//cudaMemcpy(inusrp,configs.outdata,configs.total_num_samps*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	DeliteCudaMallocHost((void**)&inusrp,configs.total_num_samps*sizeof(cuFloatComplex));
	DeliteCudaMemcpyHtoDAsync(inusrp,configs.outdata,configs.total_num_samps*sizeof(cuFloatComplex));
	//
	complex2delitearray<<<dimGrid,dimBlock>>>(inusrp,x,y,configs.total_num_samps);
	//
	//
	//float *hostPtr,*xdata=(float*)malloc(x.length*sizeof(float));
	//DeliteCudaMallocHost((void**)&hostPtr,x.length*sizeof(float));
	//DeliteCudaMemcpyDtoHAsync(hostPtr,x.data,x.length*sizeof(float));
	//memcpy(xdata,hostPtr,x.length*sizeof(float));
	//
	//free(configs.outdata);
	//
	//DeliteCudaToc("usrpstream");
}
//

void usrpstreamer(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y, double freq)
{
	//DeliteCudaTic("usrpstream");
	//
	//TODO: Set the center frequency
   	std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        //if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        configs.usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (configs.usrp->get_rx_freq()/1e6) << std::endl << std::endl;
	//
	rx_while();
	//TODO: Copy DataBack to Device
	//
	//
	//printf("X Len = %d ::: Y Len = %d .\n",x.length,y.length);
	dim3 dimBlock, dimGrid;
  int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
  // Set up grid
  // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(configs.total_num_samps/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	cuFloatComplex *inusrp;
	//cudaMalloc((void**)&inusrp,configs.total_num_samps*sizeof(cuFloatComplex));
	//cudaMemcpy(inusrp,configs.outdata,configs.total_num_samps*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	DeliteCudaMallocHost((void**)&inusrp,configs.total_num_samps*sizeof(cuFloatComplex));
	DeliteCudaMemcpyHtoDAsync(inusrp,configs.outdata,configs.total_num_samps*sizeof(cuFloatComplex));
	//
	complex2delitearray<<<dimGrid,dimBlock>>>(inusrp,x,y,configs.total_num_samps);
	//
	//
	//float *hostPtr,*xdata=(float*)malloc(x.length*sizeof(float));
	//DeliteCudaMallocHost((void**)&hostPtr,x.length*sizeof(float));
	//DeliteCudaMemcpyDtoHAsync(hostPtr,x.data,x.length*sizeof(float));
	//memcpy(xdata,hostPtr,x.length*sizeof(float));
	//
	//free(configs.outdata);
	//
	//DeliteCudaToc("usrpstream");
}
//

//
void usrpstream(int flag,cudaDeliteArrayfloat x,cudaDeliteArrayfloat y)
{
	//DeliteCudaTic("usrpstream");
	streamin_usrp();
	//TODO: Copy DataBack to Device
	//
	//
	dim3 dimBlock, dimGrid;
  int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
  // Set up grid
  // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(configs.total_num_samps/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	cuFloatComplex *inusrp;
	cudaMalloc((void**)&inusrp,configs.total_num_samps*sizeof(cuFloatComplex));
	cudaMemcpy(inusrp,configs.outdata,configs.total_num_samps*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	complex2delitearray<<<dimGrid,dimBlock>>>(inusrp,x,y,configs.total_num_samps);
	//
	cudaFree(inusrp);
	free(configs.outdata);
	//
	//DeliteCudaToc("usrpstream");
}
// This is where Comments Ended*/
//
//
__global__ void optisdrifftscale(cuFloatComplex *invec,cudaDeliteArrayfloat outrl,cudaDeliteArrayfloat outim,int fp, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		outrl.update(tid,cuCrealf(invec[tid])/fp);
		outim.update(tid,cuCimagf(invec[tid])/fp);
		tid += blockDim.x * gridDim.x;
	}
}
//
// Zero Padding
//
__global__ void zrpad(cudaDeliteArrayfloat inp,cudaDeliteArrayfloat outp,int ncol, int ocol)
{
	//int row = blockIdx.y*blockDim.y + threadIdx.y;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	for(int i=0; i<ncol;i++)
	{
		outp.update(ocol*i+index,inp.apply(ncol*i+index));
	}
}
//
// Cross-Correlation Function 1
//
void xcorr(cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim, int ftpoint)
{
	//DeliteCudaTic("xcorr");
	//
	//printf("Testing Lengths:[x1]= %d.\n [x2]= %d.\n [y1]= %d.\n [y2]= %d.\n [outrl]= %d.\n [outim]= %d.\n", x1.length, x2.length, y1.length, y2.length,outrl.length, outim.length);
	//
    	int sigLen = x1.length/ftpoint;
    	int chunk = x1.length;
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	/*
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	cuFloatComplex *ddata1,*dout1,*dout2,*ddata2;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;
	//
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	//
	cudaMalloc((void**)&ddata1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&ddata2,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout2,chunk*sizeof(cuFloatComplex));
	//
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x1,y1,ddata1,chunk);
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x2,y2,ddata2,chunk);
	//
	// Execution		
	//
	res = cufftExecC2C(plan,ddata1,dout1,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}		
	//
	res = cufftExecC2C(plan,ddata2,dout2,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	if(x1.length == x2.length)
	{
		complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,chunk);
	}
	else
	{
		printf("[SDR_DSL_INFO]$ Warning: Vector A and B must be of the same size.\n");
	}
	//
	res = cufftExecC2C(plan,ddata1,ddata2,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//optisdrifftscale<<<dimGrid,dimBlock>>>(dhout,dout,ftpoint,chunk);
	optisdrifftscale<<<dimGrid,dimBlock>>>(ddata2,outrl,outim,ftpoint,chunk);
	//
	// Releasing Computing Resources
	cudaFree(ddata1);
	cudaFree(dout1);
	cudaFree(dout2);
	cudaFree(ddata2);
	//
	cufftDestroy(plan);
	//
	*/
	//DeliteCudaToc("xcorr");
	//
}
//
// Cross-Correlation Function 2
//
void xcorr(int ftpoint, cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim)
{
	//DeliteCudaTic("xcorr");
	//	
	//
	//printf("Testing Lengths:[x1]= %d.\n [x2]= %d.\n [y1]= %d.\n [y2]= %d.\n", x1.length, x2.length, y1.length, y2.length);
    	int sigLen = x1.length/ftpoint;
    	int chunk = x1.length;
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	cuFloatComplex *ddata1,*dout1,*dout2,*ddata2;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;
	//
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	//
	cudaMalloc((void**)&ddata1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&ddata2,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout2,chunk*sizeof(cuFloatComplex));
	//
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x1,y1,ddata1,chunk);
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x2,y2,ddata2,chunk);
	//
	// Execution		
	//
	res = cufftExecC2C(plan,ddata1,dout1,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}		
	//
	res = cufftExecC2C(plan,ddata2,dout2,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,chunk);
	//
	if(x1.length == x2.length)
	{
		complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,chunk);
	}
	else
	{
		//
		printf("[SDR_DSL_INFO]$ Warning: Vector A and B must be of the same size.\n");
		if(x1.length<x2.length)
			complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,x1.length);
		else
			complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,x2.length);
	}
	//
	//
	res = cufftExecC2C(plan,ddata1,ddata2,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//optisdrifftscale<<<dimGrid,dimBlock>>>(dhout,dout,ftpoint,chunk);
	optisdrifftscale<<<dimGrid,dimBlock>>>(ddata2,outrl,outim,ftpoint,chunk);
	//
	// Releasing Computing Resources
	cudaFree(ddata1);
	cudaFree(dout1);
	cudaFree(dout2);
	cudaFree(ddata2);
	//
	cufftDestroy(plan);
	//
	//DeliteCudaToc("xcorr");
	//
}
//
void xcorr(int ftpoint, cudaDeliteArrayfloat x1, cudaDeliteArrayfloat x2, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y1, cudaDeliteArrayfloat y2, cudaDeliteArrayfloat outim, int ftp)
{
	//DeliteCudaTic("xcorr");
	//	
	//
	//printf("Testing Lengths:[x1]= %d.\n [x2]= %d.\n [y1]= %d.\n [y2]= %d.\n", x1.length, x2.length, y1.length, y2.length);
    	int sigLen = x1.length/ftpoint;
    	int chunk = x1.length;
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	cuFloatComplex *ddata1,*dout1,*dout2,*ddata2;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;
	//
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	//
	cudaMalloc((void**)&ddata1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&ddata2,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout1,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout2,chunk*sizeof(cuFloatComplex));
	//
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x1,y1,ddata1,chunk);
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x2,y2,ddata2,chunk);
	//
	// Execution		
	//
	res = cufftExecC2C(plan,ddata1,dout1,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}		
	//
	res = cufftExecC2C(plan,ddata2,dout2,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,chunk);
	//
	if(x1.length == x2.length)
	{
		complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,chunk);
	}
	else
	{
		//
		printf("[SDR_DSL_INFO]$ Warning: Vector A and B must be of the same size.\n");
		if(x1.length<x2.length)
			complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,x1.length);
		else
			complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout1,dout2,ddata1,x2.length);
	}
	//
	//
	res = cufftExecC2C(plan,ddata1,ddata2,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//optisdrifftscale<<<dimGrid,dimBlock>>>(dhout,dout,ftpoint,chunk);
	optisdrifftscale<<<dimGrid,dimBlock>>>(ddata2,outrl,outim,ftpoint,chunk);
	//
	// Releasing Computing Resources
	cudaFree(ddata1);
	cudaFree(dout1);
	cudaFree(dout2);
	cudaFree(ddata2);
	//
	cufftDestroy(plan);
	//
	//DeliteCudaToc("xcorr");
	//
}

//
// TODO: Hilbert Transform Must only return the imag value - No need to copy back real part as its the original. Mem. Optimization
//
void hilbert(cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim, int ftpoint)
{
	//DeliteCudaTic("hilbert");
	//
    	int sigLen = x.length/ftpoint;
    	int chunk = x.length;
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//	
	cuFloatComplex* h = resizeVector(getHilbertHVector(ftpoint),ftpoint,chunk);
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	cuFloatComplex *ddata,*dout,*dh;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;
	//
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	//	
	cudaMalloc((void**)&dh,chunk*sizeof(cuFloatComplex));
	cudaMemcpy(dh,h,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	cudaMalloc((void**)&ddata,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout,chunk*sizeof(cuFloatComplex));
	//cudaMalloc((void**)&dhout[i],chunk*sizeof(cuFloatComplex));
	//cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
	//
	//
	copyrtoc<<<dimGrid,dimBlock>>>(x,y,ddata,chunk);
	// Execution		
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout,dh,ddata,chunk);
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//optisdrifftscale<<<dimGrid,dimBlock>>>(dhout,dout,ftpoint,chunk);
	optisdrifftscale<<<dimGrid,dimBlock>>>(dout,outrl,outim,ftpoint,chunk);
	//
	// Releasing Computing Resources
	cudaFree(ddata);
	cudaFree(dout);
	//cudaFree(dhout);
	//cudaFree(drout);
	//free(hdata[i]);
	cufftDestroy(plan);
	cudaFree(dh);
	free(h);
	//DeliteCudaToc("hilbert");
	//
}
//
//
void hilbert2(int ftpoint, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim)
{
	//
    	int sigLen = x.length/ftpoint;
    	int chunk = x.length;
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;//, blocksPerGrid;
	//
	dimBlock.x = 1024;
	//dimBlock.y = 1024; // Uncomment for 2-D Grid
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(chunk/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	//	
	cuFloatComplex* h = resizeVector(getHilbertHVector(ftpoint),ftpoint,chunk);
	//
	cufftHandle plan;
	//int N_SIGS2 = chunk/ftpoint; // Chunk must be a multiple of ftpoint
	int n[1] = {ftpoint};
	//int sigLen = chunk/ftpoint; // Make sure chunk is multiple of 2, better done at DSL level
	//
	//
	cuFloatComplex *ddata,*dout,*dh;
	//
	// Creating cuFFT plans and sets them in streams
	//
	cufftResult res;
	//
	res = cufftPlanMany(&plan, 1, n,
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		NULL, 1,ftpoint, //advanced data layout, NULL shuts it off
    		CUFFT_C2C,sigLen);
   	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Plan create fail.\n");}
	//
	//	
	cudaMalloc((void**)&dh,chunk*sizeof(cuFloatComplex));
	cudaMemcpy(dh,h,chunk*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	//
	cudaMalloc((void**)&ddata,chunk*sizeof(cuFloatComplex));
	cudaMalloc((void**)&dout,chunk*sizeof(cuFloatComplex));
	//cudaMalloc((void**)&dhout[i],chunk*sizeof(cuFloatComplex));
	//cudaMalloc((void**)&drout[i],chunk*sizeof(cuFloatComplex));
	//
	//
	//DeliteCudaTic("hilbert");
	copyrtoc<<<dimGrid,dimBlock>>>(x,y,ddata,chunk);
	// Execution		
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_FORWARD);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	complexvector_multiply1d1d<<<dimGrid,dimBlock>>>(dout,dh,ddata,chunk);
	//
	res = cufftExecC2C(plan,ddata,dout,CUFFT_INVERSE);
	if (res != CUFFT_SUCCESS) {printf("[SDR_DSL_INFO]$ Forward transform fail.\n");}
	//
	//optisdrifftscale<<<dimGrid,dimBlock>>>(dhout,dout,ftpoint,chunk);
	optisdrifftscale<<<dimGrid,dimBlock>>>(dout,outrl,outim,ftpoint,chunk);
	//
	//DeliteCudaToc("hilbert");
	// Releasing Computing Resources
	cudaFree(ddata);
	cudaFree(dout);
	//cudaFree(dhout);
	//cudaFree(drout);
	//free(hdata[i]);
	cufftDestroy(plan);
	cudaFree(dh);
	free(h);
	//
}
//
//
void ifftx2(int SIG_LEN, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim)
{
		//DeliteCudaTic("ifftx2");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//			printf("Testing Signal Lengths: outrl=%d, outim=%d, y = %d and x= %d.\n",outrl.length,outim.length,y.length,x.length);
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_INVERSE);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			optisdrifftscale<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,outrl,outim,SIG_LEN,dataLen);
		}
		else
		{
			optisdrifftscale<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,outrl,outim,SIG_LEN,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("ifftx2");
}
//
//
void ifftx(cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim, int SIG_LEN)
{
		//DeliteCudaTic("ifftx");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//			printf("Testing Signal Lengths: outrl=%d, outim=%d, y = %d and x= %d.\n",outrl.length,outim.length,y.length,x.length);
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_INVERSE);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			optisdrifftscale<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,outrl,outim,SIG_LEN,dataLen);
		}
		else
		{
			optisdrifftscale<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,outrl,outim,SIG_LEN,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("ifftx");
}
//
//
void fftx2( int SIG_LEN, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim)
{
		//DeliteCudaTic("fftx2");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//			printf("Testing Signal Lengths: outrl=%d, outim=%d, y = %d and x= %d.\n",outrl.length,outim.length,y.length,x.length);
//			printf("Value 0 of x = %f, Value 0 of y = %f .\n",x.apply(0),y.apply(0));
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			copydelitector<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,outrl,outim,dataLen);
		}
		else
		{
			copydelitector<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,outrl,outim,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("fftx2");
}
//
//
void fftx(cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim, int SIG_LEN)
{
		//DeliteCudaTic("fftx");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
			//printf("Testing Signal Lengths: outrl=%d, outim=%d, y = %d and x= %d.\n",outrl.length,outim.length,y.length,x.length);
			cuFloatComplex *tstout = (cuFloatComplex*)malloc(x.length*sizeof(cuFloatComplex));
			//printf("N_SIGS = %d .\n",N_SIGS);
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			//
			cudaMemcpy(tstout,d_signal,dataLen*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
			/*printf("\n\n");
			for(int i = 0; i<10;i++)printf("data[%i] = (%f+j%f) , ",i,tstout[i].x,tstout[i].y);
			printf("\n\n");*/
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			copydelitector<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,outrl,outim,dataLen);
		}
		else
		{
			copydelitector<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,outrl,outim,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("fftx");
}
//
//
void ifftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat out, int SIG_LEN)
{
		//DeliteCudaTic("ifftc");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//			printf("Testing Signal Lengths: out=%d, y = %d and x= %d.\n",out.length,y.length,x.length);
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_INVERSE);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			copyctor<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,out,dataLen);
		}
		else
		{
			copyctor<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,out,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("ifftc");
}
//
void fftc(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat out, int SIG_LEN)
{
		//DeliteCudaTic("fftc");
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//			printf("Testing Signal Lengths: out=%d, y = %d and x= %d.\n",out.length,y.length,x.length);
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
			if(1 +((dataLen-1)/512) > 65535)
			{
				copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
			else
			{
				copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
			}
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
   cufftHandle plan;
   int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
//Copy FFT Results from Device to Host
//
		if(1 +((dataLen-1)/512) > 65535)
		{
			copyctor<<<dim3(65535,1,1),dim3(512,1,1)>>>(d_result,out,dataLen);
		}
		else
		{
			copyctor<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(d_result,out,dataLen);
		}
//
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
		//DeliteCudaToc("fftc");
}
//
void fftp(cppDeliteArrayfloat x,cppDeliteArrayfloat y,cppDeliteArrayfloat out, int SIG_LEN)
{
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result,*h_signal,*h_result;
    	int N_SIGS = x.length/SIG_LEN;
    	int dataLen = N_SIGS*SIG_LEN;
//
//		h_signal = getComplexSamples(x,y,dataLen);
			h_signal = (cuFloatComplex *)malloc(dataLen*sizeof(cuFloatComplex));			
			h_result = (cuFloatComplex *)malloc(dataLen*sizeof(cuFloatComplex));
//
			for(int i = 0; i< dataLen;i++)
			{
				h_signal[i] = make_cuFloatComplex((float)x.apply(i),(float)y.apply(i));
			}
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
    cudaMemcpy(d_signal, h_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
//
// Create an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
    cufftHandle plan;
    int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
// Copy FFT Results from Device to Host
    cudaMemcpy(h_result, d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
//
			//printf("Testing Signal Lengths: out=%d, y = %d and x= %d.\n",out.length,y.length,x.length);

		for(int i = 0; i< dataLen;i++)
		{
			out.update(2*i,(double)cuCrealf(h_result[i]));
			out.update(2*i+1,(double)cuCimagf(h_result[i]));
		}
/*
OK Lets See If We get this far...
*/
//
		free(h_signal);
		free(h_result);
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
}
//
void fftq(double *x,double *out, double *y, int SIG_LEN)
{
//
//Variable declarations
    	cuFloatComplex *d_signal, *d_result,*h_signal,*h_result;
    	int N_SIGS = 8192;
    	int dataLen = N_SIGS*SIG_LEN;
//
//		h_signal = getComplexSamples(x,y,dataLen);
			h_signal = (cuFloatComplex *)malloc(dataLen*sizeof(cuFloatComplex));			
			h_result = (cuFloatComplex *)malloc(dataLen*sizeof(cuFloatComplex));
//
			for(int i = 0; i< dataLen;i++)
			{
				h_signal[i] = make_cuFloatComplex((float)x[i],(float)y[i]);
			}
//
//Device Memory Allocations
    cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
//
// Copy Data from Host to Device
    cudaMemcpy(d_signal, h_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
//
// Crerate an FFT Plan Here - Needs cufftPlanMany(...) for a Batched execution
    cufftHandle plan;
    int n[1] = {SIG_LEN};
//
    cufftResult res = cufftPlanMany(&plan, 1, n,
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    NULL, 1, SIG_LEN, //advanced data layout, NULL shuts it off
    CUFFT_C2C, N_SIGS);
    if (res != CUFFT_SUCCESS) {printf("plan create fail\n");}
//
// Execute the FFT
    res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {printf("forward transform fail\n");}
//
// Copy FFT Results from Device to Host
    cudaMemcpy(h_result, d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
//
		//for(int i = 0; i< dataLen;i++)
		//{
		//	out[2*i]   =cuCrealf(h_result[i]);
		//	out[2*i+1] =cuCimagf(h_result[i]);
		//}
//
		free(h_signal);
		free(h_result);
    cudaFree(d_signal);
    cudaFree(d_result);
//
    cufftDestroy(plan);
}
//
//%%%%%%%%%%%%%%%%%%
//
void exec_c2c_dsp(cudaDeliteArrayfloat x,cudaDeliteArrayfloat y,cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat outim, int funcID,int flen)
{
	switch(funcID)
	{
		case Hilbert:
			hilbert(x,outrl,y,outim,flen);
			break;
		case Fft:
			fftx(x,y,outrl,outim,flen);
			break;
		case Ifft:
			ifftx(x,outrl,y,outim,flen);
			break;

		case XCorr:
			xcorr(x,x,outrl,y,y,outim,flen);
			break;
		
		case Conv:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Please Implement it!");
			break;
		
		case Ddc:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Please Implement it!");
			break;

		case Log10:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Please Implement it!");
			break;

		case Abs:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Please Implement it!");
			break;

		case Psd:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Please Implement it!");
			break;

		case Streamprocessor:
			
			break;
		default:
			printf("[OptiSDR_INFO]$ Module Doesn't Exist. Menu of Parallel Ops:\n");
			printf("[%d] hilbert\n[%d] fft\n[%d] ifft\n[%d] xcorr\n[%d] conv\n[%d] ddc\n[%d] Log10\n[%d] Abs\n[%d] psd\n[%d] streamprocessor\n",Hilbert,Fft,Ifft,XCorr,Conv,Ddc,Log10,Abs,Psd,Streamprocessor);
			break;
	}
}
//
void streamprocessor( cudaDeliteArrayfloat x,cudaDeliteArrayfloat y, 
			   cudaDeliteArrayfloat funcIDs, cudaDeliteArrayfloat funcPARs,
			   cudaDeliteArrayfloat outx, cudaDeliteArrayfloat outy)
{
	//
	DeliteCudaTic("streamprocessor");
	//
	//Variable declarations
    	//cuFloatComplex *d_signal, *d_result;
    	int N_SIGS = x.length/x.length;
    	int dataLen = N_SIGS*x.length;
	//
	printf("Testing Signal Lengths: outx=%d, outy=%d, y = %d and x= %d.\n",outx.length,outy.length,y.length,x.length);
	cuFloatComplex *tstout = (cuFloatComplex*)malloc(dataLen*sizeof(cuFloatComplex));
	printf("N_SIGS = %d .\n",N_SIGS);
	//
	//Device Memory Allocations
	/*cudaMalloc(&d_signal, dataLen*sizeof(cuFloatComplex));
	cudaMalloc(&d_result, dataLen*sizeof(cuFloatComplex));
	//
	// Copy Data from Host to Device
	if(1 +((dataLen-1)/512) > 65535)
	{
		copyrtoc<<<dim3(65535,1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
	}
	else
	{
		copyrtoc<<<dim3(1 +((dataLen-1)/512),1,1),dim3(512,1,1)>>>(x,y,d_signal,dataLen);
	}*/
	//
	exec_c2c_dsp(x,y,outx,outy,(int)funcIDs.apply(0),(int)funcPARs.apply(0));
	//
	//cudaMemcpy(tstout,d_signal,dataLen*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	//printf("\n\n");
	//for(int i = 0; i<10;i++)printf("data[%i] = (%f+j%f) , ",i,tstout[i].x,tstout[i].y);
	//printf("\n\n");
	//cudaFree(d_signal);
	//cudaFree(d_result);
	//
	DeliteCudaToc("streamprocessor");
}
//
// Abs of a Complex Vector
//
// Abs Kernel Definition
//
__global__ void abs_kernel(cudaDeliteArrayfloat x, cudaDeliteArrayfloat y, cudaDeliteArrayfloat out, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
	{
		out.update(tid,cuCabsf(make_cuFloatComplex(x.apply(tid),y.apply(tid))));
	}
}
// Abs Kernel instantiation
void Absv(int M, cudaDeliteArrayfloat x, cudaDeliteArrayfloat outrl, cudaDeliteArrayfloat y, cudaDeliteArrayfloat outim)
{
	//
	dim3 dimBlock, dimGrid;
  	int threadsPerBlock;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(M/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	/*printf("\nX Len = %d, Y Len = %d, RL Len = %d, IM Len = %d \n",x.length,y.length,outrl.length,outim.length);
    	cuFloatComplex *d_signal, *h_result;
    	int dataLen = x.length; // This should equal M
	h_result = (cuFloatComplex *)malloc(dataLen*sizeof(cuFloatComplex));
	//
	//Device Memory Allocations
  	cudaMalloc(&d_signal,dataLen*sizeof(cuFloatComplex));
	//
	// Copy Data from Host to Device
	copyrtoc<<<dimGrid,dimBlock>>>(x,y,d_signal,dataLen);*/
	//
	abs_kernel<<<dimGrid,dimBlock>>>(x,y,outrl,M);
	//
	/*cudaMemcpy(h_result,d_signal,dataLen*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	printf("\n[ ");
	for(int i = 0; i<10; i++)
	{
		printf("%f + j%f, ",h_result[i].x,h_result[i].y);
	}
	printf(" ]\n");
	cudaFree(d_signal);
	free(h_result);
	printf("Check if %d = %d .\n",dataLen,M);*/
}
//
// Sqrt of a Vector
//
// Kernel Def
//
__global__ void sqrt_kernel(cudaDenseVectorFloat x,cudaDenseVectorFloat out, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
	{
		out.dc_update(tid,sqrtf(x.dc_apply(tid)));
	}
}
// Sqrt Kernel Instantiation
void Sqrtv(cudaDenseVectorFloat x,cudaDenseVectorFloat out, int M)
{
	//
	dim3 dimBlock, dimGrid;
    	int threadsPerBlock;
	//
	dimBlock.x = 1024;
	//
	threadsPerBlock = dimBlock.x*dimBlock.y*dimBlock.z;
        // Set up grid
        // Here you will have to set up the variables: dimGrid and blocksPerGrid
	int grid1 = ceil(M/(float)threadsPerBlock); // For 1-D Grid
	//
	dimGrid.x = grid1;
	//
	sqrt_kernel<<<dimGrid,dimBlock>>>(x,out,M);
	//
}
//
// Parallel Streamer
//
void parallelstreamer(int xC,cudaDeliteArrayfloat* src,cudaDeliteArrayint32_t* funcs,cudaDeliteArrayfloat* outx,cudaDeliteArraydouble* pars,cudaDeliteArrayfloat* outy, int size)
{
	//
	// tic() - Similar to Octave
	//
	DeliteCudaTic("parallelstreamer");
	//
	// Testing Parameters arrangements
	//
	printf("Testing Signal Lengths: outx=%d, outy=%d, src = %d and funcs= %d.\n",outx->length,outy->length,src->length,funcs->length);
	//
	// Executing Functions
	//
	//exec_c2c_dsp(x,y,outx,outy,(int)funcIDs.apply(0),(int)funcPARs.apply(0));
	//
	// toc() - Similar to Octave
	//
	DeliteCudaToc("parallelstreamer");
}
//
//%%%%%
#endif
