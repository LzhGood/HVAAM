#ifndef __IMAGEMANAGER_H__
#define __IMAGEMANAGER_H__

#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <io.h>
#include <random>
#include <algorithm>
#include <string>

using namespace std;
//using namespace cv;

class ImageManager
{
public:
	ImageManager(const std::string &dir,const bool &needRandom = true);
	~ImageManager()
	{}


	int getNum();
	int getCurrentIndex();
	string getCurrentImageName();

	cv::Mat next();
private:
	string dirPath = "";
	int numOfImages = 0;
	vector<string> imageIndexSet;
	int imageIndex = 0;
	cv::Mat background;
	string imageName="";
};

#endif

