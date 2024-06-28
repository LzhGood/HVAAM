#include "GazeContingent.h"

GazeContingent::GazeContingent(Size s) :size(s)
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

vector<Mat> GazeContingent::getBlendingFunctions()
{
	return blendingFunctions;
}

vector<float> GazeContingent::getBandwidthSets()
{
	return bandwidthSets;
}





/*
Mat GazeContingent::update(int mouseX, int mouseY)
{
	int biasY = centerPoint.y - mouseY;
	int biasX = centerPoint.x - mouseX;
	float R;
	Mat output = Mat::zeros(Size(centerPoint.x, centerPoint.y), CV_8UC3);
	for (int x = 0; x < pyramidContainer[0].cols; x++)
	{
		for (int y = 0; y < pyramidContainer[0].rows; y++)
		{
			int x_ = biasX + x;
			int y_ = biasY + y;
			R = resolutionMap.at<float>(y_, x_);
			if (R <= bandwidthSets[0] && R >= bandwidthSets[1])//2
			{
				output.at<Vec3b>(y, x)[0] = float(blendingFunctions[1].at<float>(y_, x_)*pyramidContainer[0].at<Vec3b>(y, x)[0] + (1 - blendingFunctions[1].at<float>(y_, x_))*pyramidContainer[1].at<Vec3b>(y, x)[0]);
				output.at<Vec3b>(y, x)[1] = float(blendingFunctions[1].at<float>(y_, x_)*pyramidContainer[0].at<Vec3b>(y, x)[1] + (1 - blendingFunctions[1].at<float>(y_, x_))*pyramidContainer[1].at<Vec3b>(y, x)[1]);
				output.at<Vec3b>(y, x)[2] = float(blendingFunctions[1].at<float>(y_, x_)*pyramidContainer[0].at<Vec3b>(y, x)[2] + (1 - blendingFunctions[1].at<float>(y_, x_))*pyramidContainer[1].at<Vec3b>(y, x)[2]);

			}
			else if (R <= bandwidthSets[1] && R >= bandwidthSets[2])
			{
				output.at<Vec3b>(y, x)[0] = float(blendingFunctions[2].at<float>(y_, x_)*pyramidContainer[1].at<Vec3b>(y, x)[0] + (1 - blendingFunctions[2].at<float>(y_, x_))*pyramidContainer[2].at<Vec3b>(y, x)[0]);
				output.at<Vec3b>(y, x)[1] = float(blendingFunctions[2].at<float>(y_, x_)*pyramidContainer[1].at<Vec3b>(y, x)[1] + (1 - blendingFunctions[2].at<float>(y_, x_))*pyramidContainer[2].at<Vec3b>(y, x)[1]);
				output.at<Vec3b>(y, x)[2] = float(blendingFunctions[2].at<float>(y_, x_)*pyramidContainer[1].at<Vec3b>(y, x)[2] + (1 - blendingFunctions[2].at<float>(y_, x_))*pyramidContainer[2].at<Vec3b>(y, x)[2]);

			}
			else if (R <= bandwidthSets[2] && R >= bandwidthSets[3])
			{
				output.at<Vec3b>(y, x)[0] = float(blendingFunctions[3].at<float>(y_, x_)*pyramidContainer[2].at<Vec3b>(y, x)[0] + (1 - blendingFunctions[3].at<float>(y_, x_))*pyramidContainer[3].at<Vec3b>(y, x)[0]);
				output.at<Vec3b>(y, x)[1] = float(blendingFunctions[3].at<float>(y_, x_)*pyramidContainer[2].at<Vec3b>(y, x)[1] + (1 - blendingFunctions[3].at<float>(y_, x_))*pyramidContainer[3].at<Vec3b>(y, x)[1]);
				output.at<Vec3b>(y, x)[2] = float(blendingFunctions[3].at<float>(y_, x_)*pyramidContainer[2].at<Vec3b>(y, x)[2] + (1 - blendingFunctions[3].at<float>(y_, x_))*pyramidContainer[3].at<Vec3b>(y, x)[2]);

			}
			else if (R <= bandwidthSets[3] && R >= bandwidthSets[4])
			{
				output.at<Vec3b>(y, x)[0] = float(blendingFunctions[4].at<float>(y_, x_)*pyramidContainer[3].at<Vec3b>(y, x)[0] + (1 - blendingFunctions[4].at<float>(y_, x_))*pyramidContainer[4].at<Vec3b>(y, x)[0]);
				output.at<Vec3b>(y, x)[1] = float(blendingFunctions[4].at<float>(y_, x_)*pyramidContainer[3].at<Vec3b>(y, x)[1] + (1 - blendingFunctions[4].at<float>(y_, x_))*pyramidContainer[4].at<Vec3b>(y, x)[1]);
				output.at<Vec3b>(y, x)[2] = float(blendingFunctions[4].at<float>(y_, x_)*pyramidContainer[3].at<Vec3b>(y, x)[2] + (1 - blendingFunctions[4].at<float>(y_, x_))*pyramidContainer[4].at<Vec3b>(y, x)[2]);

			}
			else if (R <= bandwidthSets[4] && R >= bandwidthSets[5])
			{
				output.at<Vec3b>(y, x)[0] = float(blendingFunctions[5].at<float>(y_, x_)*pyramidContainer[4].at<Vec3b>(y, x)[0] + (1 - blendingFunctions[5].at<float>(y_, x_))*pyramidContainer[5].at<Vec3b>(y, x)[0]);
				output.at<Vec3b>(y, x)[1] = float(blendingFunctions[5].at<float>(y_, x_)*pyramidContainer[4].at<Vec3b>(y, x)[1] + (1 - blendingFunctions[5].at<float>(y_, x_))*pyramidContainer[5].at<Vec3b>(y, x)[1]);
				output.at<Vec3b>(y, x)[2] = float(blendingFunctions[5].at<float>(y_, x_)*pyramidContainer[4].at<Vec3b>(y, x)[2] + (1 - blendingFunctions[5].at<float>(y_, x_))*pyramidContainer[5].at<Vec3b>(y, x)[2]);

			}
		}
	}

	return output;
}
*/


