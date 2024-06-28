#include "ImageManager.h"

ImageManager::ImageManager(const std::string &dir, const bool &needRandom) :dirPath(dir)
{
	_finddatai64_t file;
	long long HANDLE = 0;
	HANDLE = _findfirst64(string(dirPath.assign(dir) + "*.png").c_str(), &file);
	imageIndexSet.push_back(string(file.name));
	numOfImages++;
	int a = _findnexti64(HANDLE, &file);
	do
	{
		imageIndexSet.push_back(string(file.name));
		numOfImages++;
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);


	background = cv::Mat(480, 640, CV_8UC3);
	for (int i = 0; i < background.rows; i++)
	{
		for (int j = 0; j < background.cols; j++)
		{
			background.at<cv::Vec3b>(i, j)[0] = 105;
			background.at<cv::Vec3b>(i, j)[1] = 105;
			background.at<cv::Vec3b>(i, j)[2] = 105;
		}
	}

	if (needRandom == true)
	{
		shuffle(imageIndexSet.begin(), imageIndexSet.end(), default_random_engine(6738));
	}

	//for (auto &it : imageIndexSet)
	//{
	//	cout << it << endl;
	//}
}

cv::Mat ImageManager::next()
{
	imageName = imageIndexSet[imageIndex];
	cv::Mat source= cv::imread(dirPath + imageName, cv::IMREAD_COLOR);
	if (source.empty())
	{
		cerr << "Can not open Image: " << dirPath+imageIndexSet[imageIndex] << endl;
		exit(0);
	}
	imageIndex++;

	//1920->1280;1080->960
	float r = 480.0/source.rows;
	float c = 640.0/source.cols;

	if (r<=c)
	{
		cv::resize(source, source, cv::Size(int(source.cols*r), 480), 0,0, cv::INTER_LINEAR);
	}
	else
	{
		cv::resize(source, source, cv::Size(640, int(source.rows*c)),0,0 , cv::INTER_LINEAR);
	}

	int roiX = (640 - source.cols) / 2;
	int roiY = (480 - source.rows) / 2;
	cv::Mat backCopy = background.clone();
	cv::Mat imageROI = backCopy(cv::Rect(roiX, roiY, source.cols, source.rows));
	cv::Mat tmp = cv::Mat::ones(source.size(), CV_8UC1);
	source.copyTo(imageROI, tmp);

	//imshow("aaa",background);
	return backCopy;
}

int ImageManager::getNum()
{
	return numOfImages;
}

int ImageManager::getCurrentIndex()
{
	return imageIndex;
}

string ImageManager::getCurrentImageName()
{
	return imageName;
}