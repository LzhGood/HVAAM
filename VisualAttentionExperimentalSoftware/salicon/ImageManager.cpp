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


	background = Mat(960, 1280, CV_8UC3);
	for (int i = 0; i < background.rows; i++)
	{
		for (int j = 0; j < background.cols; j++)
		{
			background.at<Vec3b>(i, j)[0] = 105;
			background.at<Vec3b>(i, j)[1] = 105;
			background.at<Vec3b>(i, j)[2] = 105;
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

inline Mat ImageManager::next()
{
	imageName = imageIndexSet[imageIndex];
	Mat source=imread(dirPath + imageName, IMREAD_COLOR);
	if (source.empty())
	{
		cerr << "Can not open Image: " << dirPath+imageIndexSet[imageIndex] << endl;
		exit(0);
	}
	imageIndex++;

	//1920->1280;1080->960
	float r = 960.0/source.rows;
	float c = 1280.0/source.cols;

	if (r<=c)
	{
		resize(source, source, Size(int(source.cols*r), 960), 0,0,INTER_LINEAR);
	}
	else
	{
		resize(source, source, Size(1280, int(source.rows*c)),0,0 ,INTER_LINEAR);
	}

	int roiX = (1280 - source.cols) / 2;
	int roiY = (960 - source.rows) / 2;
	Mat backCopy = background.clone();
	Mat imageROI = backCopy(Rect(roiX, roiY, source.cols, source.rows));
	Mat tmp = Mat::ones(source.size(), CV_8UC1);
	source.copyTo(imageROI, tmp);

	//imshow("aaa",background);
	return backCopy;
}

inline int ImageManager::getNum()
{
	return numOfImages;
}

inline int ImageManager::getCurrentIndex()
{
	return imageIndex;
}

inline string ImageManager::getCurrentImageName()
{
	return imageName;
}