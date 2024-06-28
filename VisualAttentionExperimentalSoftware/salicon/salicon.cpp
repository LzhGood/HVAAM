#include "GazeContingent.h"
#include "ImageManager.h"
#include <Windows.h>

using namespace std;
using namespace cv;
string path = "D:\\Image\\";
int mouse_X = 0;
int mouse_Y = 0;

void on_mouse(int event, int x, int y, int flags, void *ustc);
int main()
{
	namedWindow("Output");
	GazeContingent g(Size(2560,1920));
	ImageManager m(path, true);
	g.initGaze();
	g.initForNewImage(m.next());

	setMouseCallback("Output",on_mouse,0);
	int key = '1';
	while (true)
	{
		Mat output = g.update(mouse_X, mouse_Y);
		imshow("Output", output);
		key = waitKey(20);
		if (key == 'Q')
		{
			break;
		}
		else if (key == 'N')
		{
			g.initForNewImage(m.next());
		}
	}
	//cout << f[4] << endl;
	return 1;
}

void on_mouse(int event, int x, int y, int flags, void *ustc)
{
	if (event == EVENT_MOUSEMOVE)
	{
		mouse_X = x;
		mouse_Y = y;
	}
}


