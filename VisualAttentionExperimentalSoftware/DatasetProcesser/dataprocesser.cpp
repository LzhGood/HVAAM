#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "ImageManager.h"
#include <math.h>
#include <stdio.h>
#include <Windows.h>

//#define IMAGEOUT
//#define IMAGE
//#define FLIP
//#define MEAN
//#define AUC

using namespace std;
//using namespace cv;

const int index = 3;
const int fixationThreshold = 12;
const int fixationTime = 100;

string fixationListPath = "D:\\Fixation\\";
string output = "D:\\FixationMap\\";
cv::Mat generateSaliencyMap(vector<string> &fixationsPath, int index, cv::Mat &gaussianKernal);
cv::Mat getGussianKernal(int kernalSize, double sigma, bool normalize);
cv::Mat generateFixationMap(vector<string> &fixationsPath, int index);
void drawGazePoints(vector<string> &fixationsPath, int index, cv::Mat &source);
void drawGazePointsNew(vector<string> &fixationsPath, int index, cv::Mat &source);
cv::Mat drawHeatMap(cv::Mat frame, cv::Mat heatmap);
double generateGaussianNoise(double mu, double sigma);
cv::Mat addGaussianNoise(cv::Mat &image);
cv::Mat coarseDropout(cv::Mat &input, int size, int num);
cv::Mat generateSaliencyMapByPoint(int x, int y, cv::Mat &gaussianKernal);

int main()
{
#ifdef AUC

	string valPath = "D:\\LZH\\AUC\\val\\FixationMap\\";

	vector<string> predictionList;
	predictionList.push_back("D:\\LZH\\AUC\\val\\Prediction\\1\\");
	predictionList.push_back("D:\\LZH\\AUC\\val\\Prediction\\2\\");
	predictionList.push_back("D:\\LZH\\AUC\\val\\Prediction\\3\\");
	predictionList.push_back("D:\\LZH\\AUC\\val\\Prediction\\4\\");

	vector<string> classesList;
	classesList.push_back("panel\\");
	classesList.push_back("vehicle\\");
	classesList.push_back("web\\");
  
	_finddatai64_t file;
	long long HANDLE = 0;
	ofstream txtWriter;

	vector<int> indexList;

	for (int i = 0; i < classesList.size(); i++)
	{
		indexList.clear();
		HANDLE = _findfirst64(string(valPath + classesList[i] + "*.png").c_str(), &file);
		indexList.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
		int a = _findnexti64(HANDLE, &file);
		do
		{
			indexList.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
		} while (_findnexti64(HANDLE, &file) != -1);
		_findclose(HANDLE);

		for (int j = 0; j < predictionList.size(); j++)
		{
			vector<float> tprListSum;
			vector<float> fprListSum;
			for (int n = 0; n <indexList.size(); n++)
			{
				Mat g = imread(valPath+classesList[i] + to_string(indexList[n]) + ".png", IMREAD_GRAYSCALE);
				Mat p = imread(predictionList[j] + classesList[i] + to_string(indexList[n]) + ".png", IMREAD_GRAYSCALE);

				//ROC

				vector<float> tprList;
				vector<float> fprList;
				for (int threshold = 255; threshold >=0; threshold--)
				{
					int tp = 0;
					int fp = 0;
					int tn = 0;
					int fn = 0;

					for (int r = 0; r < g.rows; r++)
					{
						uchar *g_pointer = g.ptr<uchar>(r);
						uchar *p_pointer = p.ptr<uchar>(r);

						for (int c = 0; c < g.cols; c++)
						{
							if (p_pointer[c] >= threshold)
							{
								if (g_pointer[c] > 100)
								{
									tp++;
								}
								else
								{
									fp++;
								}
							}
							else
							{
								if (g_pointer[c] > 100)
								{
									fn++;
								}
								else
								{
									tn++;
								}
							}
						}
					}

					tprList.push_back(tp*1.0/(tp+fn));
					fprList.push_back(fp*1.0/(fp+tn));
				}
				
				//写入文件
				txtWriter.open(predictionList[j] + classesList[i] + "ROC\\" + "ROC-" +to_string(indexList[n]) + ".txt", ios::trunc);
				for (int txtIndex = 0; txtIndex < tprList.size(); txtIndex++)
				{
					txtWriter << fprList[txtIndex] << " " << tprList[txtIndex] << endl;
					
					if (n > 0)
					{
						tprListSum[txtIndex] += tprList[txtIndex];
						fprListSum[txtIndex] += fprList[txtIndex];
					}
					else
					{
						tprListSum.push_back(tprList[txtIndex]);
						fprListSum.push_back(fprList[txtIndex]);
					}
				}
				txtWriter.close();
				cout << "Processing " + predictionList[j] + classesList[i] + to_string(indexList[n]) + ".png" << endl;
			}
			//写入均值ROC
			txtWriter.open(predictionList[j] + classesList[i] + "ROC\\" + "ROC_Mean.txt", ios::trunc);
			for (int txtIndex = 0; txtIndex < tprListSum.size(); txtIndex++)
			{
				txtWriter << fprListSum[txtIndex]/ indexList.size() << " " << tprListSum[txtIndex]/ indexList.size() << endl;
			}
			txtWriter.close();
		}
	}

	return 1;
#endif // AUC
		   
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
	compression_params.push_back(9);

	string pathSource = "D:\\LZH\\Image_\\newData\\forClipData\\cliped\\Source\\";
	string pathSaliency = "D:\\LZH\\Image_\\newData\\forClipData\\cliped\\SaliencyMap\\";
	string pathFixation = "D:\\LZH\\Image_\\newData\\forClipData\\cliped\\FixationMap\\";

	string pathSource_ = "D:\\humanComputerInteraction\\result1\\Source\\";
	string pathSaliency_ = "D:\\humanComputerInteraction\\result1\\Saliency\\";
	string pathFixation_ = "D:\\LZH\\Image_\\newData\\forClipData\\val\\FixationMap\\";

	string pathOut = "D:\\humanComputerInteraction\\Old_Image\\LZH\\Image_\\newData\\forClipData\\Source\\";

#ifdef FLIP
	vector<int> imageIndexSetSource; 
	vector<int> imageIndexSetSaliency;
	vector<int> imageIndexSetFixation;

	int in =1;
	cv::Mat source = cv::imread(pathSource_ + to_string(in) + ".png");
	cv::Mat saliency = cv::imread(pathSaliency_ + to_string(in) + ".png");
	cv::Mat source1 = source.clone();
	cv::Mat b = saliency.clone();
	cv::Mat c = drawHeatMap(source1, b);
	cv::Mat tmp = source.clone();
	cv::GaussianBlur(tmp, tmp, cv::Size(25, 25), 3.0, 3.0);
	cv::imshow("Source", source);
	//imshow("Gaussian", tmp);
	//imshow("Saliency", saliency); 
	cv::imshow("Heatmap", c);
	//imwrite("D:\\" + to_string(39) + ".png", tmp, compression_params);
	cv::waitKey(0);
	return 1;

	/*Source*/
	_finddatai64_t file;
	long long HANDLE = 0;
	HANDLE = _findfirst64(string(pathSource_ + "*.png").c_str(), &file);
	imageIndexSetSource.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	int a = _findnexti64(HANDLE, &file);
	do
	{
		imageIndexSetSource.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);

	/*Saliency*/
	HANDLE = _findfirst64(string(pathSaliency + "*.png").c_str(), &file);
	imageIndexSetSaliency.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	a = _findnexti64(HANDLE, &file);
	do
	{
		imageIndexSetSaliency.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);
	
	/*Fixation*/
	HANDLE = _findfirst64(string(pathFixation + "*.png").c_str(), &file);
	imageIndexSetFixation.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	a = _findnexti64(HANDLE, &file);
	do
	{
		imageIndexSetFixation.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);




	vector<int> index;
	for (int i = 0; i < 381; i++)
	{
		index.push_back(i);
	}
	::shuffle(index.begin(), index.end(), default_random_engine(5211));
	for (int n = 0; n < 100; n++)
	{
		MoveFileA((pathSource + to_string(index[n]) + ".png").c_str(), (pathSource_ + to_string(index[n]) + ".png").c_str());
		MoveFileA((pathSaliency + to_string(index[n]) + ".png").c_str(), (pathSaliency_ + to_string(index[n]) + ".png").c_str());
		MoveFileA((pathFixation + to_string(index[n]) + ".png").c_str(), (pathFixation_ + to_string(index[n]) + ".png").c_str());

		remove((pathSource + to_string(index[n] + 381) + ".png").c_str());
		remove((pathSource + to_string(index[n] + 381*2) + ".png").c_str());
		remove((pathSource + to_string(index[n] + 381*3) + ".png").c_str());
		remove((pathSource + to_string(index[n] + 381*4) + ".png").c_str());

		remove((pathSaliency + to_string(index[n] + 381) + ".png").c_str());
		remove((pathSaliency + to_string(index[n] + 381 * 2) + ".png").c_str());
		remove((pathSaliency + to_string(index[n] + 381 * 3) + ".png").c_str());
		remove((pathSaliency + to_string(index[n] + 381 * 4) + ".png").c_str());

		remove((pathFixation + to_string(index[n] + 381) + ".png").c_str());
		remove((pathFixation + to_string(index[n] + 381 * 2) + ".png").c_str());
		remove((pathFixation + to_string(index[n] + 381 * 3) + ".png").c_str());
		remove((pathFixation + to_string(index[n] + 381 * 4) + ".png").c_str());
	}

	return 1;
	/*Sort*/
	std::sort(imageIndexSetSource.begin(), imageIndexSetSource.end());
	std::sort(imageIndexSetSaliency.begin(), imageIndexSetSaliency.end());
	std::sort(imageIndexSetFixation.begin(), imageIndexSetFixation.end());

	/*for (int i = 1143; i < 1524; i++)
	{
		rename((pathSource + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathSource + to_string(i) + ".png").c_str());
		rename((pathSaliency + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathSaliency + to_string(i) + ".png").c_str());
		rename((pathFixation + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathFixation + to_string(i) + ".png").c_str());
	}
	return 1;*/

	for (int i = 0; i < 381; i++)
	{
		cv::Mat tmp = cv::imread(pathSource + to_string(i) + ".png");
		//GaussianBlur(tmp, tmp, cv::Size(5, 5), 11.0, 11.0);
		//tmp = addGaussianNoise(tmp);
		coarseDropout(tmp,3,1000);
		//imshow("AA",tmp);
		//waitKey(0);
		//return 1;
		cv::imwrite(pathSource + to_string(1524 + i) + ".png", tmp, compression_params);
		CopyFileA((pathSaliency + to_string(i) + ".png").c_str(), (pathSaliency + to_string(1524 + i) + ".png").c_str(), true);
		CopyFileA((pathFixation + to_string(i) + ".png").c_str(), (pathFixation + to_string(1524 + i) + ".png").c_str(), true);

	}

	return 1;

	/*for (int i = 0; i < imageIndexSetSource.size(); i++)
	{
		rename((pathSource + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathSource + to_string(i) + ".png").c_str());
		rename((pathSaliency + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathSaliency + to_string(i) + ".png").c_str());
		rename((pathFixation + to_string(imageIndexSetSource[i]) + ".png").c_str(), (pathFixation + to_string(i) + ".png").c_str());
	}
	return 1;	*/
	set<int> deleteIndex;
	deleteIndex.insert(200);
	deleteIndex.insert(206);
	deleteIndex.insert(219);
	deleteIndex.insert(220);
	deleteIndex.insert(221);
	deleteIndex.insert(478);
	deleteIndex.insert(417);
	deleteIndex.insert(419);
	deleteIndex.insert(468);
	deleteIndex.insert(473);
	deleteIndex.insert(474);
	deleteIndex.insert(400);
	deleteIndex.insert(432);
	deleteIndex.insert(431);
	deleteIndex.insert(42);
	deleteIndex.insert(194);


	int key = 'A';
	for (int i = 0; i < imageIndexSetSource.size(); i++)
	{
		key = 'A';
		cv::Mat source = cv::imread(pathSource + to_string(imageIndexSetSource[i]) + ".png");
		cv::Mat saliency = cv::imread(pathSaliency + to_string(imageIndexSetSaliency[i]) + ".png");
		cv::Mat frmae = source.clone();
		cv::Mat sa = saliency.clone();
		cv::Mat he=drawHeatMap(frmae,sa);
		cv::imshow("Source",source);
		cv::imshow("Saliency", saliency);
		cv::imshow("HeatMap", he);


		key= cv::waitKey(0);

		if (key == ' ')
		{

		}
		else if (key == 'L')
		{
			deleteIndex.insert(i);
			std::cout << "Delete: " << i << endl;
		}
		else if (key == 'Q')
		{
			break;
		}
		else if (key == 'M')
		{
			std::cout <<"Move: "<< i << endl;
		}
	}

	for (auto setIndex = deleteIndex.begin(); setIndex != deleteIndex.end(); setIndex++)
	{
		remove((pathSource + to_string(*setIndex) + ".png").c_str());
		remove((pathSaliency + to_string(*setIndex) + ".png").c_str());
		remove((pathFixation + to_string(*setIndex) + ".png").c_str());
	}

	return 1;

	::shuffle(imageIndexSetSource.begin(), imageIndexSetSource.end(), default_random_engine(1234));
	for (int n = 0; n < 600; n++)
	{
		//MoveFileA((path + to_string(imageIndexSetSource[n]) + ".png").c_str(), (pathOut + to_string(imageIndexSetSource[n]) + ".png").c_str());
		//remove((path + to_string(imageIndexSetSource[0]) + ".png").c_str());
	}
	//for (int n = 0; n < imageIndexSetSource.size(); n++)
	//{
	//	Mat tmp = imread(path + to_string(imageIndexSetSource[n]) + ".png");
	//	flip(tmp, tmp, 0);//>0,y;x=0;<0,xy;
	//	//rotate(tmp, tmp, 1);
	//	//imshow("aa", tmp);
	//	//waitKey(0);
	//	imwrite(pathOut + to_string(1500 + n) + ".png", tmp, compression_params);
	//}

	return 1;
#endif // FLIP

#ifdef MEAN
	vector<int> imageIndexSetSource;
	int numOfImages = 0;
	_finddatai64_t file;
	long long HANDLE = 0;
	string path = "D:\\LZH\\Image_\\Source\\";
	HANDLE = _findfirst64(string(path + "*.png").c_str(), &file);
	imageIndexSetSource.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
	numOfImages++;
	int a = _findnexti64(HANDLE, &file);
	do
	{
		imageIndexSetSource.push_back(stoi(string(file.name).substr(0, string(file.name).find_last_of("."))));
		numOfImages++;
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);


	unsigned __int64 r = 0;
	unsigned __int64 g = 0;
	unsigned __int64 b = 0;
	std::sort(imageIndexSetSource.begin(), imageIndexSetSource.end());
	for (int n = 0; n < imageIndexSetSource.size(); n++)
	{
		cout << "Image: " << n << endl;
		Mat tmp = imread(path + to_string(imageIndexSetSource[n]) + ".png");//BGR
		for (int x = 0; x < tmp.cols; x++)
		{
			for (int y = 0; y < tmp.rows; y++)
			{
				b += tmp.at<Vec3b>(y,x)[0];
				g += tmp.at<Vec3b>(y,x)[1];
				r += tmp.at<Vec3b>(y,x)[2];
			}
		}
	}
	cout << "R: " << r / (640.0*480.0*imageIndexSetSource.size()) << endl;
	cout << "G: " << g / (640.0*480.0*imageIndexSetSource.size()) << endl;
	cout << "B: " << b / (640.0*480.0*imageIndexSetSource.size()) << endl;
	return 1;
#endif // MEAN


#ifdef IMAGE
	_finddatai64_t file;
	long long HANDLE = 0;
	HANDLE = _findfirst64(string(path + "*").c_str(), &file);
	int a = _findnexti64(HANDLE, &file);
	int i = 0;
	do
	{
		if (string(file.name) == ".." || string(file.name) == ".")
		{
			continue;
		}
		Mat tmp = imread(path+string(file.name));
		cout << file.name << endl;
		imwrite(pathOut + to_string(i) + ".png", tmp);
		i++;
	} while (_findnexti64(HANDLE, &file) != -1);
	_findclose(HANDLE);
	return 1;
#endif // IMAGE

#ifdef IMAGEOUT
	ImageManager m(path,false);
	Mat tmp;
	for (int i = 0; i < m.getNum(); i++)
	{
		tmp = m.next();
		imwrite(pathOut+m.getCurrentImageName(),tmp);
	}
	return 1;
#else
	
	cv::Mat kernal = getGussianKernal(131, 23, true);//sigma越小越集中，越大越分散;61-13;101-21
												 //cout << kernal << endl;
	vector<string> rawDataPathList;
	vector<string> rawDataHandledPathList;
	string rawDataPrefix = "D:\\humanComputerInteraction\\matplot\\RawData\\"; //RawData
	string rawDataHandledPrefix = "D:\\humanComputerInteraction\\matplot\\RawData_handled\\";//RawData_handled

	rawDataHandledPathList.push_back(rawDataHandledPrefix + "HouSongtao_output\\");
	rawDataHandledPathList.push_back(rawDataHandledPrefix + "LiJiakun_output\\");

	rawDataPathList.push_back(rawDataPrefix + "HouSongtao\\");
	rawDataPathList.push_back(rawDataPrefix + "LiJiakun\\");

	for (int i = 0; i < 160; i++)
	{
		cv::Mat a = generateFixationMap(rawDataHandledPathList, i);
		cv::Mat b = generateSaliencyMap(rawDataHandledPathList, i, kernal);
		cv::Mat source1 = cv::imread(pathOut + to_string(i) + ".png");//640*480 source image
		cv::Mat source1_copy = source1.clone();
		cv::Mat source1_copy_2 = source1.clone();

		drawGazePointsNew(rawDataHandledPathList, i, source1);
		drawGazePoints(rawDataPathList, i, source1_copy_2);//包含试验数据点坐标偏移功能
		cv::Mat c = drawHeatMap(source1_copy, b);
		cv::imshow("处理数据后的图像", source1);
		cv::imshow("处理数据后的HeatMap", c);
		cv::imshow("处理数据后的SaliencyMap", b);
		cv::imshow("没处理数据的图像", source1_copy_2);
		//cv::imshow("QQQ",a);
		//imshow("Source2", source2);
		//cout << "Process image: " << i << endl;
		//imwrite("D:\\LZH\\tmp\\SaliencyMap\\"+to_string(i)+".png",b, compression_params);
		//imwrite("D:\\LZH\\tmp\\FixationMap\\" + to_string(i) + ".png", a, compression_params);

		int key = cv::waitKey(0);
//		cv::imwrite("D:\\LZH\\tmp\\Completed_3_to_1\\Completed_3_to_1\\fixation_" + to_string(i) + ".png", a, compression_params);

		if (key == 'S')
		{
			cv::imwrite("D:\\humanComputerInteraction\\HeatMap\\"+to_string(i)+".png", c, compression_params);
			cv::imwrite("D:\\humanComputerInteraction\\SaliencyMap\\" + to_string(i) + ".png", b, compression_params);
		}
	}

	return 1;
#endif // IMAGEOUT
}


void drawGazePoints(vector<string> &fixationsPath, int index, cv::Mat &source)
{
	int i = 0;
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;

		cv::Point2f avr(0, 0);
		int num = 0;
		int sumTime = 0;
		while (getline(fileReader, line))
		{
			istringstream is(line);
			string tmp;
			cv::Point point;
			//is >> tmp;//name
			is >> tmp;//x
			point.x = int((stof(tmp)-329)*0.5);//转为640X480分辨率下的fixation
			is >> tmp;//y
			point.y = int((stof(tmp)-112)*0.5);
			is >> tmp;
			if (stof(tmp) > 2000)
			{
				continue;
			}
			if (point.x >= 640 || point.x < 0 || point.y >= 480 || point.y < 0)
			{
				continue;
			}
			i++;
			//if (i !=1)
			//{
			//	circle(source, point, 3, Scalar(0, 255, 0),-1);
			//	continue;
			//}
			circle(source, point, 2, cv::Scalar(255, 0, 255),-1);//bgr
		}
		fileReader.close();
	}
}

cv::Mat getGussianKernal(int kernalSize, double sigma,bool normalize)
{
	const double PI = 3.14159265358979323846;
	cv::Mat gussianKernal(kernalSize, kernalSize, CV_64FC1);
	double sigma2 = 2*sigma*sigma;
	int m = (kernalSize - 1) / 2;
	float sum = 0;
	int x_ = 0;
	int y_ = 0;

	for (int x = 0; x < kernalSize; x++)
	{
		for (int y = 0; y < kernalSize; y++)
		{
			x_ = x - m;
			y_ = y - m;
			float tmp = exp(-(x_*x_ + y_*y_) / sigma2) / (PI*sigma2);
			gussianKernal.ptr<double>(x)[y] = tmp;
			sum = sum + tmp;
		}
	}
	if (normalize == true)
	{
		for (int x = 0; x < kernalSize; x++)
		{
			for (int y = 0; y < kernalSize; y++)
			{
				gussianKernal.at<double>(x, y) = gussianKernal.at<double>(x, y) / sum;
			}
		}
	}
	return gussianKernal;
}

cv::Mat generateFixationMap(vector<string> &fixationsPath, int index)
{
	cv::Mat sum8 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;
		string tmp;
		cv::Point point;
		bool isFirstIn = true;
		while (getline(fileReader, line))
		{
			if (isFirstIn == true)
			{
				isFirstIn = false;
				continue;
			}
			istringstream is(line);

			is >> tmp;//x
			point.x = stoi(tmp);//转为640X480分辨率下的fixation
			is >> tmp;//y
			point.y = stoi(tmp);
			if (point.x < 0 || point.x >=640 || point.y < 0 || point.y >=480)
			{
				continue;
			}
			//sum8.at<uchar>(point.y, point.x) = 255;
			cv::circle(sum8, point, 3, cv::Scalar(255, 0, 255), -1);
		}
		fileReader.close();
	}
	return sum8;
}

/*
Mat generateFixationMap(vector<string> &fixationsPath, int index)
{
	Mat sum8 = Mat::zeros(Size(640, 480), CV_8UC1);
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;
		while (getline(fileReader, line))
		{
			istringstream is(line);
			string tmp;
			Point point;
			is >> tmp;//name
			is >> tmp;//x
			//point.x = int(stof(tmp)*0.3333);//转为640X480分辨率下的fixation
			point.x = int(stof(tmp) - 327);
			is >> tmp;//y
			//point.y = int(stof(tmp)*0.4444);
			point.y = int(stof(tmp) - 113);
			if (point.x >= 640 || point.x < 0 || point.y >= 480 || point.y < 0)
			{
				continue;
			}

			sum8.at<uchar>(point.y, point.x) = 1;
		}
		fileReader.close();
	}
	return sum8;
}
*/

cv::Mat generateSaliencyMap(vector<string> &fixationsPath, int index, cv::Mat &gaussianKernal)
{
	cv::Mat sum = cv::Mat::zeros(cv::Size(640, 480), CV_64FC1);
	cv::Mat sum8 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	int biasX = 0;
	int biasY = 0;
	int x_, y_;
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;
		string tmp;
		cv::Point point;
		bool isFirstIn = true;
		while (getline(fileReader, line))
		{
			if (isFirstIn == true)
			{
				isFirstIn = false;
				continue;
			}
			istringstream is(line);

			is >> tmp;//x
			point.x = stoi(tmp);
			is >> tmp;//y
			point.y = stoi(tmp);
			if (point.x < 0 || point.x >= 640 || point.y < 0 || point.y >= 480)
			{
				continue;
			}

			biasX = point.x - (gaussianKernal.size[0] - 1) * 0.5;
			biasY = point.y - (gaussianKernal.size[0] - 1) * 0.5;

			for (int x = 0; x < gaussianKernal.size[0]; x++)
			{
				for (int y = 0; y < gaussianKernal.size[0]; y++)
				{
					x_ = biasX + x;
					y_ = biasY + y;
					if (x_ < 0 || x_ >= 640 || y_ < 0 || y_ >= 480)
					{
						continue;
					}
					sum.at<double>(y_, x_) += gaussianKernal.at<double>(y, x) * 100;
				}
			}

		}
		fileReader.close();
	}

	double minVal, maxVal;
	cv::minMaxIdx(sum, &minVal, &maxVal);
	double diff = maxVal - minVal;
	for (int i = 0; i < 640; i++)
	{
		for (int j = 0; j < 480; j++)
		{
			sum8.at<uchar>(j, i) = sum.at<double>(j, i) / diff * 255;
		}
	}
	return sum8;
}


cv::Mat generateSaliencyMapByPoint(int x,int y, cv::Mat &gaussianKernal)
{
	cv::Mat sum = cv::Mat::zeros(cv::Size(320, 240), CV_64FC1);
	cv::Mat sum8 = cv::Mat::zeros(cv::Size(320, 240), CV_8UC1);
	int biasX = 0;
	int biasY = 0;
	int x_, y_;
	cv::Point point;

	point.x = x;
	point.y = y;


	biasX = point.x - (gaussianKernal.size[0] - 1) * 0.5;
	biasY = point.y - (gaussianKernal.size[0] - 1) * 0.5;

	for (int x = 0; x < gaussianKernal.size[0]; x++)
	{
		for (int y = 0; y < gaussianKernal.size[0]; y++)
		{
			x_ = biasX + x;
			y_ = biasY + y;
			if (x_ < 0 || x_ >= 320 || y_ < 0 || y_ >= 240)
			{
				continue;
			}
			sum.at<double>(y_, x_) += gaussianKernal.at<double>(y, x) * 100;
		}
	}



	double minVal, maxVal;
	cv::minMaxIdx(sum, &minVal, &maxVal);
	double diff = maxVal - minVal;
	for (int i = 0; i < 320; i++)
	{
		for (int j = 0; j < 240; j++)
		{
			sum8.at<uchar>(j, i) = sum.at<double>(j, i) / diff * 255;
		}
	}
	return sum8;
}


/*
Mat generateSaliencyMap(vector<string> &fixationsPath, int index, Mat &gaussianKernal)
{
	Mat sum = Mat::zeros(Size(640, 480), CV_64FC1);
	Mat sum8 = Mat::zeros(Size(640, 480), CV_8UC1);
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;
		while (getline(fileReader, line))
		{
			istringstream is(line);
			string tmp;
			Point point;
			is >> tmp;//name
			is >> tmp;//x
			//point.x = int(stof(tmp)*0.3333);//转为640X480分辨率下的fixation
			point.x = int(stof(tmp) - 327);
			is >> tmp;//y
			//point.y = int(stof(tmp)*0.4444);
			point.y = int(stof(tmp) - 113);
			if (point.x >= 640 || point.x < 0 || point.y >= 480 || point.y < 0)
			{
				continue;
			}

			int biasX = point.x - (gaussianKernal.size[0] - 1) / 2;
			int biasY = point.y - (gaussianKernal.size[0] - 1) / 2;
			int x_, y_;
			for (int x = 0; x < gaussianKernal.size[0]; x++)
			{
				for (int y = 0; y < gaussianKernal.size[0]; y++)
				{
					x_ = biasX + x;
					y_ = biasY + y;
					if (x_ < 0 || x_ >= 640 || y_ < 0 || y_ >= 480)
					{
						continue;
					}

					sum.at<double>(y_, x_) += gaussianKernal.at<double>(y, x) * 1000;
				}
			}
		}
		fileReader.close();
	}

	double minVal, maxVal;
	minMaxIdx(sum, &minVal, &maxVal);
	double diff = maxVal - minVal;
	for (int i = 0; i < 640; i++)
	{
		for (int j = 0; j < 480; j++)
		{
			sum8.at<uchar>(j, i) = sum.at<double>(j, i) / diff * 255;
		}
	}
	return sum8;
}
*/

void drawGazePointsNew(vector<string> &fixationsPath, int index, cv::Mat &source)
{
	for (auto path : fixationsPath)
	{
		ifstream fileReader;
		fileReader.open(path + to_string(index) + ".txt", ios::in);
		if (fileReader.is_open() == false)
		{
			continue;
		}

		string line;
		bool isFirst = true;
		while (getline(fileReader, line))
		{
			if (isFirst == true)
			{
				isFirst = false;
				continue;
			}
			istringstream is(line);
			string tmp;
			cv::Point point;
			is >> tmp;//x
			point.x = stoi(tmp);
			is >> tmp;//y
			point.y = stoi(tmp);
			if (point.x < 0 || point.x >= 640 || point.y < 0 || point.y >= 480)
			{
				continue;
			}

			cv::circle(source, point, 3, cv::Scalar(255, 0, 255), -1);
		}
		fileReader.close();
	}
}

cv::Mat drawHeatMap(cv::Mat fram, cv::Mat heatmap)
{
	cv::Mat frame = fram.clone();
	cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);//COLORMAP_JET COLORMAP_HOT
	cv::Mat overlay = frame.clone();
	double alpha= 0.5;
	rectangle(overlay, cv::Rect(0, 0, frame.cols, frame.rows), cv::Scalar(255, 0, 0), -1);
	//addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);
	addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame);
	return frame;
}


//mu高斯函数的偏移，sigma高斯函数的标准差
double generateGaussianNoise(double mu, double sigma)
{
	//定义小值，numeric_limits<double>::min()是函数,返回编译器允许的double型数最小值
	const double epsilon = (std::numeric_limits<double>::min)();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量x
	if (!flag) return z1 * sigma + mu;
	//构造随机变量
	double u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量x
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
	return z0 * sigma + mu;
}

cv::Mat addGaussianNoise(cv::Mat &image)
{
	cv::Mat result = image.clone();
	int channels = image.channels();
	int rows = image.rows, cols = image.cols * image.channels();
	//判断图像连续性
	if (result.isContinuous()) cols = rows * cols, rows = 1;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//添加高斯噪声
			int val = result.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.6) * 32;
			if (val < 0) val = 0;
			if (val > 255) val = 255;
			result.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return result;
}

cv::Mat coarseDropout(cv::Mat &input,int size,int num)
{
	for (int i = 0; i < num; i++)
	{
		int x = rand() % (input.cols);
		int y = rand() % (input.rows);
		
		int x_ = 0;
		int y_ = 0;
		for (int r = 0; r < size; r++)
		{
			for (int c = 0; c < size; c++)
			{
				y_ = y + (r - (size - 1) * 0.5);
				x_ = x + (c - (size - 1) * 0.5);
				if (x_ < 0 || x_ >= input.cols || y_ < 0 || y_ >= input.rows)
				{
					continue;
				}
				input.at<cv::Vec3b>(y_, x_)[0] = 0;
				input.at<cv::Vec3b>(y_, x_)[1] = 0;
				input.at<cv::Vec3b>(y_, x_)[2] = 0;
			}
		}
	}

	return input;
}






