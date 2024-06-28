#include <Windows.h>
#include <thread>
#include <mutex>
#include "ImageManager.h"
#include <math.h>

#include <chrono>
#include <future>
#include "GazeContingent.h"

using namespace std;
using namespace cv;

string participatorName = "001";

volatile unsigned long timeCounter = 0;
volatile bool exitFlag = false;

string path;
string fixationSavePath;
std::mutex mut;
ofstream file;
POINT mouse;
int randomSeed = 15;
int experimentPart = 1;

int mouseEventX = 0;
int mouseEventY = 0;
bool isMiddle = false;

int timer();
int getMouseXY();
void MouseMove(int x, int y);
void recordMouseEyePoint(string &name, const DWORD &time);
void on_mouse(int event, int x, int y, int flags, void *ustc);


int main(int argc, char* argv[])
{
	cout << "Please input your ID: ";
	cin >> participatorName;
	cout << "Input random seed: ";
	cin >> randomSeed;
	cout << "Choose Experiment part: ";
	cin >> experimentPart;

	path = "D:\\humanComputerInteraction\\Old_Image\\LZH\\Image_\\" + to_string(experimentPart) + "\\";
	fixationSavePath = "D:\\humanComputerInteraction\\Old_Image\\LZH\\ExperimentData\\" + participatorName + "\\";

	string command = "mkdir " + fixationSavePath;
	system(command.c_str());
	Mat background = imread("D:\\humanComputerInteraction\\Projects\\Projects\\smartEyeConnector\\salicon\\background.png");
	Mat forPause = imread("D:\\humanComputerInteraction\\Projects\\Projects\\smartEyeConnector\\salicon\\english_start.png");;
	Mat forMiddle = imread("D:\\humanComputerInteraction\\Projects\\Projects\\smartEyeConnector\\salicon\\forMiddle.png");

	thread t(timer);
	thread x(getMouseXY);

	ImageManager m(path, true, randomSeed);
	Mat image=m.next();
	Mat image_backup = image.clone();
	bool isPaused = false;
	string currentName;
	GazeContingent gaze(Size(2560, 1920));
	gaze.initGaze();


	namedWindow("BackGround", WINDOW_FULLSCREEN);
	moveWindow("BackGround", 0, 0);
	namedWindow("EyeTracker", CV_WINDOW_AUTOSIZE);
	moveWindow("EyeTracker", 320, 95);
	//moveWindow("EyeTracker", 0, 0);
	//setMouseCallback("EyeTracker", on_mouse, 0);
	//setWindowProperty("EyeTracker",CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);

	RECT rect;
	rect.left = 329;
	rect.right = 1609;
	rect.top = 112;
	rect.bottom = 1072;

	while (true)
	{
		imshow("BackGround", background);
		imshow("EyeTracker", forPause);
		//ClipCursor(&rect);
		if (waitKey(40) == ' ')
		{
			break;
		}
	}

	int num = m.getNum();
	int counter = 1;
	int key=1;

	gaze.initForNewImage(image);
	imshow("BackGround", background);

	mouseEventX = 640;
	mouseEventY = 480;
	MouseMove(969, 592);
	mut.lock();
	timeCounter = 0;
	mut.unlock();
	DWORD startTime = GetTickCount();
	bool displayFlag = false;
	DWORD a;
	while(true)
	{
		if (isPaused == false && isMiddle == false)
		{
			image = gaze.update(mouse.x - 329, mouse.y - 112);
			//image = gaze.update(mouseEventX, mouseEventY);
			recordMouseEyePoint(m.imageName, (GetTickCount() - startTime));
		}

		imshow("EyeTracker", image);//TODO 并不一定每次都需要显示，可以适当降低图片刷新频率，提高帧率
		key = waitKey(1);//TODO 或者我可以将显示放到一个线程里，然后鼠标位置采集放到一个线程里去

		if (key == ' ')//pause
		{
			key = 1;
			isPaused = !isPaused;
			if (isPaused == true)
			{
				image = forPause;
				//清除该图片下已经保存的Fixations
				ofstream file;
				currentName = m.imageName;
				file.open(fixationSavePath + currentName.substr(0, currentName.find_last_of(".")) + ".txt", ios::trunc);
				file.close();
			}
			else
			{
				mut.lock();
				timeCounter = 0;
				mut.unlock();
				MouseMove(969, 592);
			}
		}
		else if(key == 27)//exit
		{
			key = 1;
			exitFlag = true;
			t.join();
			x.join();
			return -1;
		}
		else if (key == 'S')
		{
			imwrite("D:\\humanComputerInteraction\\Old_Image\\LZH\\Image_\\06.png", image);
		}
		
		if (timeCounter >= 8 && isPaused == false)//next image
		{
			if (counter >= num)
			{
				break;
			}
			counter++;

			gaze.initForNewImage(m.next());

			isMiddle = true;
			image = forMiddle;

			if (counter % 50 == 0)
			{
				image = forPause;
				isPaused = true;
			}

			mut.lock();
			timeCounter = 0;
			mut.unlock();
		}
		
		if (isMiddle == true && timeCounter >= 2)
		{
			MouseMove(969, 592);
			isMiddle = false;
			mut.lock();
			timeCounter = 0;
			mut.unlock();
			startTime = GetTickCount();
		}

	}
	
	exitFlag = true;
	t.join();
	x.join();
	return 0;
}

int getMouseXY()
{
	DWORD a;
	while (true)
	{
		GetCursorPos(&mouse);
		if (exitFlag == true)
		{
			return 0;
		}
	}
}

inline void recordMouseEyePoint(string &name, const DWORD &time)
{
	file.open(fixationSavePath + name.substr(0, name.find_last_of(".")) + ".txt", ios::app);
	//file << mouseEventX << " " << mouseEventY << " " << time << endl;
	file << mouse.x << " " << mouse.y << " " << time << endl;
	file.close();
}

int timer()
{
	while (true)
	{
		this_thread::sleep_for(chrono::milliseconds(500));
		mut.lock();
		timeCounter++;
		mut.unlock();

		if (exitFlag == true)
		{
			return 0;
		}
	}

}

inline void MouseMove(int x, int y)
{
	double fScreenWidth = ::GetSystemMetrics(SM_CXSCREEN) - 1;
	double fScreenHeight = ::GetSystemMetrics(SM_CYSCREEN) - 1;
	double fx = x*(65535.0f / fScreenWidth);
	double fy = y*(65535.0f / fScreenHeight);
	INPUT Input = { 0 };
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
	Input.mi.dx = fx;
	Input.mi.dy = fy;
	SendInput(1,&Input,sizeof(INPUT));
}

void on_mouse(int event, int x, int y, int flags, void *ustc)
{
	if (event == EVENT_MOUSEMOVE)
	{
		mouseEventX = x;
		mouseEventY = y;
	}
}






