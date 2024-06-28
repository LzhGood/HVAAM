#pragma once
#ifndef __GAZECONTINGENT_H__
#define __GAZECONTINGENT_H__
#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <io.h>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

class GazeContingent
{
public:
	GazeContingent(Size s);
	~GazeContingent(){}
	int initGaze();
	int initForNewImage(Mat input);

	vector<Mat> getBlendingFunctions();
	vector<float> getBandwidthSets();

	Mat& update(const int &mouseX,const int &mouseY);
private:
	Size size;
	Point2f centerPoint;
	vector<Mat> blendingFunctions;
	vector<float> bandwidthSets;
	float standardDeviation = 0.248f;
	vector<Mat> pyramidContainer;
	Mat resolutionMap;
	Mat output;

	float calculateBlendFunc(float x, float y, int layerIndex);

	float alpha = 2.5f;
	float p = 7.5f;
};


inline GazeContingent::GazeContingent(Size s) :size(s)
{
	centerPoint.x = size.width *0.5;
	centerPoint.y = size.height *0.5;
}

inline int GazeContingent::initGaze()
{
	for (int layer = 1; layer < 6; layer++)
	{
		bandwidthSets.push_back(standardDeviation * sqrt(-1 * log(0.5) / (pow(2, 2 * layer - 7))));
	}
	bandwidthSets.push_back(0.0f);


	resolutionMap = Mat::zeros(size, CV_32FC1);
	for (int y = 0; y < size.height; y++)
	{
		float *r = resolutionMap.ptr<float>(y);
		for (int x = 0; x < size.width; x++)
		{
			r[x] = alpha / (sqrtf(powf(x - centerPoint.x, 2) + powf(y - centerPoint.y, 2)) / p + alpha);
		}
	}


	blendingFunctions.push_back(Mat::zeros(size, CV_32FC1));
	for (int layer = 2; layer < 7; layer++)
	{
		Mat blendMat = Mat::zeros(size, CV_32FC1);
		for (int y = 0; y < size.height; y++)
		{
			float *b = blendMat.ptr<float>(y);

			for (int x = 0; x < size.width; x++)
			{
				b[x] = calculateBlendFunc(x, y, layer);
			}
		}

		blendingFunctions.push_back(blendMat);

	}

	return	1;
}

inline float GazeContingent::calculateBlendFunc(float x, float y, int layerIndex)
{
	//float theta = sqrtf(powf(x - centerPoint.x, 2) + powf(y - centerPoint.y, 2)) / p;
	//float R = alpha / (theta+alpha);

	float R = resolutionMap.at<float>(y, x);
	//TODO: generate a resolution map for check out
	switch (layerIndex)
	{
	case 2:
	{
		if (R <= bandwidthSets[1])
		{
			return 0.0f;
		}
		else if (R >= bandwidthSets[0])
		{
			return 1.0f;
		}

		float Ti = exp(-0.5*pow(pow(2, -1)*R / standardDeviation, 2));//2
		float Ti_1 = exp(-0.5*pow(pow(2, -2)*R / standardDeviation, 2));//1
		return (0.5 - Ti) / (Ti_1 - Ti);
	}
	case 3:
	{
		if (R <= bandwidthSets[2])
		{
			return 0.0f;
		}
		else if (R >= bandwidthSets[1])
		{
			return 1.0f;
		}

		float Ti = exp(-0.5*pow(pow(2, 0)*R / standardDeviation, 2));//3
		float Ti_1 = exp(-0.5*pow(pow(2, -1)*R / standardDeviation, 2));//2
		return (0.5 - Ti) / (Ti_1 - Ti);
	}
	case 4:
	{
		if (R <= bandwidthSets[3])
		{
			return 0.0f;
		}
		else if (R >= bandwidthSets[2])
		{
			return 1.0f;
		}

		float Ti = exp(-0.5*pow(pow(2, 1)*R / standardDeviation, 2));//4
		float Ti_1 = exp(-0.5*pow(pow(2, 0)*R / standardDeviation, 2));//3
		return (0.5 - Ti) / (Ti_1 - Ti);
	}
	case 5:
	{
		if (R <= bandwidthSets[4])
		{
			return 0.0f;
		}
		else if (R >= bandwidthSets[3])
		{
			return 1.0f;
		}

		float Ti = exp(-0.5*pow(pow(2, 2)*R / standardDeviation, 2));//5
		float Ti_1 = exp(-0.5*pow(pow(2, 1)*R / standardDeviation, 2));//4
		return (0.5 - Ti) / (Ti_1 - Ti);
	}
	case 6:
	{
		if (R <= bandwidthSets[5])
		{
			return 0.0f;
		}
		else if (R >= bandwidthSets[4])
		{
			return 1.0f;
		}

		float Ti = 0;//6
		float Ti_1 = exp(-0.5*pow(pow(2, 2)*R / standardDeviation, 2));//5
		return (0.5 - Ti) / (Ti_1 - Ti);
	}
	default:
		return -1000000;
	}
}

inline vector<Mat> GazeContingent::getBlendingFunctions()
{
	return blendingFunctions;
}

inline vector<float> GazeContingent::getBandwidthSets()
{
	return bandwidthSets;
}

inline Mat& GazeContingent::update(const int &mouseX, const int &mouseY)
{
	if (mouseX < 0 || mouseX > centerPoint.x || mouseY <0 || mouseY > centerPoint.y)
	{
		return output;
	}

	int biasY = centerPoint.y - mouseY;
	int biasX = centerPoint.x - mouseX;
	float R = 0.0f;
	float blendTmp = 0.0f;
	int x_, y_;
	output = Mat::zeros(Size(centerPoint.x, centerPoint.y), CV_8UC3);
	for (int y = 0; y < pyramidContainer[0].rows; y++)
	{
		y_ = biasY + y;
		uchar *s = output.ptr<uchar>(y);
		float *r = resolutionMap.ptr<float>(y_);
		for (int x = 0; x < pyramidContainer[0].cols; x++)
		{
			x_ = biasX + x;
			R = r[x_];
			if (R <= bandwidthSets[0] && R >= bandwidthSets[1])//2
			{
				uchar *tmp1 = pyramidContainer[0].ptr<uchar>(y);
				uchar *tmp2 = pyramidContainer[1].ptr<uchar>(y);
				float *b = blendingFunctions[1].ptr<float>(y_);
				blendTmp = b[x_];

				s[x * 3 + 0] = blendTmp*tmp1[x * 3 + 0] + (1 - blendTmp)*tmp2[x * 3 + 0];
				s[x * 3 + 1] = blendTmp*tmp1[x * 3 + 1] + (1 - blendTmp)*tmp2[x * 3 + 1];
				s[x * 3 + 2] = blendTmp*tmp1[x * 3 + 2] + (1 - blendTmp)*tmp2[x * 3 + 2];

			}
			else if (R <= bandwidthSets[1] && R >= bandwidthSets[2])
			{
				uchar *tmp1 = pyramidContainer[1].ptr<uchar>(y);
				uchar *tmp2 = pyramidContainer[2].ptr<uchar>(y);
				float *b = blendingFunctions[2].ptr<float>(y_);
				blendTmp = b[x_];

				s[x * 3 + 0] = blendTmp*tmp1[x * 3 + 0] + (1 - blendTmp)*tmp2[x * 3 + 0];
				s[x * 3 + 1] = blendTmp*tmp1[x * 3 + 1] + (1 - blendTmp)*tmp2[x * 3 + 1];
				s[x * 3 + 2] = blendTmp*tmp1[x * 3 + 2] + (1 - blendTmp)*tmp2[x * 3 + 2];

			}
			else if (R <= bandwidthSets[2] && R >= bandwidthSets[3])
			{
				uchar *tmp1 = pyramidContainer[2].ptr<uchar>(y);
				uchar *tmp2 = pyramidContainer[3].ptr<uchar>(y);
				float *b = blendingFunctions[3].ptr<float>(y_);
				blendTmp = b[x_];

				s[x * 3 + 0] = blendTmp*tmp1[x * 3 + 0] + (1 - blendTmp)*tmp2[x * 3 + 0];
				s[x * 3 + 1] = blendTmp*tmp1[x * 3 + 1] + (1 - blendTmp)*tmp2[x * 3 + 1];
				s[x * 3 + 2] = blendTmp*tmp1[x * 3 + 2] + (1 - blendTmp)*tmp2[x * 3 + 2];

			}
			else if (R <= bandwidthSets[3] && R >= bandwidthSets[4])
			{
				uchar *tmp1 = pyramidContainer[3].ptr<uchar>(y);
				uchar *tmp2 = pyramidContainer[4].ptr<uchar>(y);
				float *b = blendingFunctions[4].ptr<float>(y_);
				blendTmp = b[x_];

				s[x * 3 + 0] = blendTmp*tmp1[x * 3 + 0] + (1 - blendTmp)*tmp2[x * 3 + 0];
				s[x * 3 + 1] = blendTmp*tmp1[x * 3 + 1] + (1 - blendTmp)*tmp2[x * 3 + 1];
				s[x * 3 + 2] = blendTmp*tmp1[x * 3 + 2] + (1 - blendTmp)*tmp2[x * 3 + 2];

			}
			else if (R <= bandwidthSets[4] && R >= bandwidthSets[5])
			{
				uchar *tmp1 = pyramidContainer[4].ptr<uchar>(y);
				uchar *tmp2 = pyramidContainer[5].ptr<uchar>(y);
				float *b = blendingFunctions[5].ptr<float>(y_);
				blendTmp = b[x_];

				s[x * 3 + 0] = blendTmp*tmp1[x * 3 + 0] + (1 - blendTmp)*tmp2[x * 3 + 0];
				s[x * 3 + 1] = blendTmp*tmp1[x * 3 + 1] + (1 - blendTmp)*tmp2[x * 3 + 1];
				s[x * 3 + 2] = blendTmp*tmp1[x * 3 + 2] + (1 - blendTmp)*tmp2[x * 3 + 2];

			}
		}
	}

	circle(output,Point(mouseX,mouseY), 30, Scalar(0, 0, 255),3);
	return output;//TODO :output zhizhen,youhua
}

inline int GazeContingent::initForNewImage(Mat input)
{
	if (input.empty())
	{
		cerr << "Input Image is Empty!" << endl;
		return -1;
	}

	pyramidContainer.clear();
	pyramidContainer.push_back(input.clone());
	Mat tmp;

	for (int i = 0; i < 6; i++)
	{
		pyrDown(pyramidContainer[i], tmp, Size(pyramidContainer[i].cols / 2, pyramidContainer[i].rows / 2));
		pyramidContainer.push_back(tmp);
	}
	for (int i = 1; i < pyramidContainer.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			pyrUp(pyramidContainer[i], tmp, Size(pyramidContainer[i].cols * 2, pyramidContainer[i].rows * 2));
			pyramidContainer[i] = tmp;
		}
	}

	//imshow("Original", pyramidContainer[0]);
	//imshow("AA", pyramidContainer[5]);
	//waitKey(0);
	return 1;
}

#endif
