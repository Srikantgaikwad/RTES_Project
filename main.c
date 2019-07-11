/**********************************************************************
*@Filename:Final Project.c
*
*@Description:Implemented different tasks to Implement Virtual goal keeper..
*
*@Author:Nagarjuna Reddy Kovuru and Srikant Gaikwad
*@Date:05/03/2018
**********************************************************************/
// Include files
//***********************************************************************************
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/param.h>
#include <sys/time.h>
#include <errno.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//***********************************************************************************
//Macros
//***********************************************************************************
#define NSEC_PER_SECOND (1000000000)
//***********************************************************************************
//Globalk Variables
//***********************************************************************************
using namespace cv;
using namespace std;
/* Different Thread ID's*/
pthread_t Frame_Thread, ObjectDetect_Thread, GoalKeeper_Thread, Sequencer_Thread, VideoOutput_Thread;
/* Sched Attributes */
pthread_attr_t Frame_Sched_Attr, ObjectDetect_Sched_Attr, GoalKeeper_Sched_Attr, Sequencer_Shed_Attr, VideoOutput_Sched_Attr;
/* Scheduling Parameter structure*/
struct sched_param Frame_param, ObjectDetection_param, GoalKeeper_param, Main_param, Sequencer_param, VideoOutput_param;
/* Time parameters */
static struct timespec Frame_Start_Time = {0, 0}, ObjectDetection_Start_Time = {0, 0}, GoalKeeper_Start_Time = {0, 0}, VideoOuput_Start_Time = {0, 0};
static struct timespec Frame_Stop_Time = {0, 0}, ObjectDetection_Stop_Time = {0, 0}, GoalKeeper_Stop_Time = {0, 0}, VideoOuput_Stop_Time = {0, 0};
static struct timespec Frame_Diff = {0,0}, ObjectDetection_Diff = {0,0}, GoalKeeper_Diff = {0,0}, VideoOuput_Diff = {0,0};
/* Mutexes*/
pthread_mutex_t FrameCapture_Mutex, Coordinate_Mutex,VideoOutput_Mutex;
/* Semaphores */
static sem_t Coordinate_Sem, GoalKeeper_Sem;
/* Output Windows*/
char ObjectDetetion_Window[] = "Object Detection", GoalKeeper_Wndow[] = "GoalKeeper Movement";
/* Coordinates */
int X_Coordinate, Y_Coordinate;
/* Area Variable s*/
const int MAXIMUM_NUMBER_OF_OBJECTS = 5, MINIMUM_AREA = 60*60;
/* device */
int Traj_Count = 0, dev=0;
/* Frame */
IplImage* frame_original;
/* Image Capture */
CvCapture* cap;
bool Answer = false;
static int VideoOutput_Count = 0;
vector<Mat> Images;
/**** Thread Iteration Counts **/
#ifdef TIME_FRAME
int Frame_Count = 0;
#endif
#ifdef TIME_OBJECTDETECTION
int ObjectDetection_Count = 0;
#endif
#ifdef TIME_GOALKEEPER
int GoalKeeper_Count = 0;
#endif
#ifdef TIME_VIDEOOUTPUT
int VideoOutput_Count = 0;
#endif
/******************************************************************//****
* @brief diff_time()
* This function gets the time difference between start time and stop time
*
***********************************************************************/
int diff_time(struct timespec *stop_time, struct timespec *start_time, struct timespec *diff_time)
{
int diff_sec=stop_time->tv_sec - start_time->tv_sec;
int diff_nsec=stop_time->tv_nsec - start_time->tv_nsec;
if(diff_sec >= 0)
{
if(diff_nsec >= 0)
{
diff_time->tv_sec=diff_sec;
diff_time->tv_nsec=diff_nsec;
}
else
{
diff_time->tv_sec=diff_sec-1;
diff_time->tv_nsec=NSEC_PER_SECOND+diff_nsec;
}
}
else
{
if(diff_nsec >= 0)
{
diff_time->tv_sec=diff_sec;
diff_time->tv_nsec=diff_nsec;
}
else
{
diff_time->tv_sec=diff_sec-1;
diff_time->tv_nsec=NSEC_PER_SECOND+diff_nsec;
}
}
return(0);
}
/******************************************************************//****
* @brief Display_Scheduler()
* This function prints Scheduler
*
***********************************************************************/
void Display_Scheduler(void)
{
int sched_policy;
/* Get Scheduling Policy */
sched_policy = sched_getscheduler( getpid() );
/* Check the policy */
switch(sched_policy)
{
case SCHED_FIFO:
printf("Scheduling Policy Is SCHED_FIFO\n");
break;
case SCHED_OTHER:
printf("Scheduling Policy Is SCHED_OTHER\n");
break;
case SCHED_RR:
printf("Scheduling Policy Is SCHED_RR\n");
break;
default:
printf("Unknown Schedulimg Policy\n");
break;
}
}
/******************************************************************//****
* @brief Frame_Capture()
* This function Captures Frames continuously
*
***********************************************************************/
void *Frame_Capture(void * param)
{
printf("[FRAME_CAPTURE] Started Task\n");
while(1)
{
#ifdef TIME_FRAME
if(Frame_Count == 0)
{
clock_gettime(CLOCK_REALTIME, &Frame_Start_Time);
printf("[FRAME_CAPTURE] Count %d Start Time: Sec= %ld and Nsec= %ld \n",Frame_Count, Frame_Start_Time.tv_sec, Frame_Start_Time.tv_nsec);
}
#endif
/* Acquire The Mutex */
pthread_mutex_lock(&FrameCapture_Mutex);
frame_original=cvQueryFrame(cap);
pthread_mutex_unlock(&FrameCapture_Mutex);
#ifdef TIME_FRAME
Frame_Count++;
if(Frame_Count == 700)
{
clock_gettime(CLOCK_REALTIME, &Frame_Stop_Time);
printf("[FRAME_CAPTURE] Count %d Stop Time: Sec= %ld and Nsec= %ld \n",Frame_Count, Frame_Stop_Time.tv_sec, Frame_Stop_Time.tv_nsec);
diff_time(&Frame_Stop_Time, &Frame_Start_Time, &Frame_Diff);
printf("[FRAME_CAPTURE] Time Difference: Sec= %ld : Nsec= %ld \n", Frame_Diff.tv_sec, Frame_Diff.tv_nsec);
exit(0);
}
#endif
}
}
/******************************************************************//****
* @brief Object_Detection()
* This function detcts object and findsout the centroid
*
***********************************************************************/
void *Object_Detection(void * param)
{
printf("[OBJECT_DETECTION] Started Task\n");
vector<Vec4i> Image_Hierarchy;
Moments Image_Memont;
stringstream Coordinate_X, Coordinate_Y, Coordinate_Final;
Mat HSV_Image, Final_Image, Erode_Element, Dilate_Element, Temp_Image;
vector< vector<Point> > Image_Contours;
int Low_H = 34, High_H = 80, Low_S = 50, High_S = 220, Low_V = 50, High_V = 200;
double Area_Cal;
while(1)
{
sem_wait(&Coordinate_Sem);
#ifdef TIME_OBJECTDETECTION
if(ObjectDetection_Count == 0)
{
clock_gettime(CLOCK_REALTIME, &ObjectDetection_Start_Time);
printf("[OBJECT_DETECTION] Count %d Start Time: Sec= %ld Nsec= %ld \n",ObjectDetection_Count, ObjectDetection_Start_Time.tv_sec, ObjectDetection_Start_Time.tv_nsec);
}
#endif
/* Acquire LLock */
pthread_mutex_lock(&FrameCapture_Mutex);
Mat mat_frame(frame_original);
pthread_mutex_unlock(&FrameCapture_Mutex);
medianBlur(mat_frame, mat_frame, 3);
/* Change Color */
cvtColor(mat_frame, HSV_Image, CV_BGR2HSV);
inRange(HSV_Image, Scalar(Low_H, Low_S, Low_V), Scalar(High_H, High_S, High_V), Final_Image);
/* Apply Filters */
Erode_Element = getStructuringElement( MORPH_RECT,Size(3,3));
Dilate_Element = getStructuringElement( MORPH_RECT,Size(8,8));
erode(Final_Image,Final_Image,Erode_Element);
erode(Final_Image,Final_Image,Erode_Element);
dilate(Final_Image,Final_Image,Dilate_Element);
dilate(Final_Image,Final_Image,Dilate_Element);
Final_Image.copyTo(Temp_Image);
/* get Image Contors */
findContours(Temp_Image,Image_Contours,Image_Hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
/* check fro Size */
if (Image_Hierarchy.size() > 0)
{
int numObjects = Image_Hierarchy.size();
/* Check whether the object is less than maximum */
if(numObjects<MAXIMUM_NUMBER_OF_OBJECTS){
for (int index = 0; index >= 0; index = Image_Hierarchy[index][0])
{
Image_Memont = moments((cv::Mat)Image_Contours[index]);
Area_Cal = Image_Memont.m00;
if(Area_Cal>MINIMUM_AREA)
{
drawContours(mat_frame,Image_Contours,-1,CV_RGB(255,0,0),3);
pthread_mutex_lock(&Coordinate_Mutex);
X_Coordinate = (Image_Memont.m10/Area_Cal);
Y_Coordinate = (Image_Memont.m01/Area_Cal);
Answer = true;
pthread_mutex_unlock(&Coordinate_Mutex);
(Coordinate_X << X_Coordinate);
(Coordinate_Y << Y_Coordinate);
Coordinate_Final << "(" << Coordinate_X.str() << "," << Coordinate_Y.str() << ")";
circle(mat_frame,Point((Image_Memont.m10/Area_Cal),(Image_Memont.m01/Area_Cal)),5,CV_RGB(255,0,0));
putText(mat_frame,Coordinate_Final.str(),Point((Image_Memont.m10/Area_Cal),(Image_Memont.m01/Area_Cal)),5,1,CV_RGB(255,0,0));
Coordinate_Final.str(std::string());
Coordinate_X.str(std::string());
Coordinate_Y.str(std::string());
}
}
}
}
/* Acquire Mutex */
pthread_mutex_lock(&VideoOutput_Mutex);
Images.push_back(mat_frame);
VideoOutput_Count++;
pthread_mutex_unlock(&VideoOutput_Mutex);
imshow( ObjectDetetion_Window, mat_frame);
char q = cvWaitKey(1);
#ifdef TIME_OBJECTDETECTION
ObjectDetection_Count++;
if(ObjectDetection_Count == 700)
{
clock_gettime(CLOCK_REALTIME, &ObjectDetection_Stop_Time);
printf("[OBJECT_DETECTION] Count: %d Stop Time: Sec= %ld and Nsec= %ld \n",ObjectDetection_Count, ObjectDetection_Stop_Time.tv_sec, ObjectDetection_Stop_Time.tv_nsec);
diff_time(&ObjectDetection_Stop_Time, &ObjectDetection_Start_Time, &ObjectDetection_Diff);
printf("[OBJECT_DETECTION] Difference Time: Sec= %ld and Nsec= %ld \n", ObjectDetection_Diff.tv_sec, ObjectDetection_Diff.tv_nsec);
exit(0);
}
#endif
}
}
/******************************************************************//****
* @brief GoalKeeper_Movement()
* This function Moves goal keeper according to the centroid calculated
*
***********************************************************************/
void *GoalKeeper_Movement(void * param)
{
printf("[GOAL_KEEPER_MOVEMENT] Started Task\n");
int GoalKeeper_Y_Axis = 480, GoalKeeper_X_Axis;
stringstream Coordinate_X,Coordinate_Y, Coordinate_Final;
Mat drawing=Mat::Zeros( 480, 1281, CV_8UC3 );
Point2i Initial_Point(0,0), Final_Point(0,0);
Mat GoalKeeper_Window,Trajectory_Window;
GoalKeeper_Window = drawing(Rect(0,0,640,480));
Trajectory_Window = drawing(Rect(641,0,640,480));
line(drawing,Point(641,0),Point(641,480),CV_RGB(255,0,0),2);
int Traj_Count =0;
bool Temp=false;
while(1)
{
/* wait on semaphore */
sem_wait(&GoalKeeper_Sem);
#ifdef TIME_GOALKEEPER
if(GoalKeeper_Count == 0)
{
clock_gettime(CLOCK_REALTIME, &GoalKeeper_Start_Time);
printf("[GOAL_KEEPER_MOVEMENT] Count: %d Start Time: Sec= %ld and Nsec= %ld \n",GoalKeeper_Count, GoalKeeper_Start_Time.tv_sec, GoalKeeper_Start_Time.tv_nsec);
}
#endif
/* Acquire the lock */
pthread_mutex_lock(&Coordinate_Mutex);
GoalKeeper_X_Axis = X_Coordinate;
GoalKeeper_Y_Axis = Y_Coordinate;
Temp = Answer;
pthread_mutex_unlock(&Coordinate_Mutex);
if(Traj_Count == 0 && Temp == true)
{
Initial_Point.x = GoalKeeper_X_Axis;
Initial_Point.y = GoalKeeper_Y_Axis;
Traj_Count++;
}
else if(Traj_Count > 0 && Temp == true)
{
Final_Point.x = GoalKeeper_X_Axis;
Final_Point.y = GoalKeeper_Y_Axis;
line(Trajectory_Window,Initial_Point,Final_Point,CV_RGB(255,0,0),4,8);
Initial_Point.x = Final_Point.x;
Initial_Point.y = Final_Point.y;
}
else
{
}
if(GoalKeeper_X_Axis < 20)
{
GoalKeeper_X_Axis = 20;
}
else if(GoalKeeper_X_Axis > 620)
{
GoalKeeper_X_Axis = 620;
}
else
{
}
line(GoalKeeper_Window,Point(GoalKeeper_X_Axis-20,480),Point(GoalKeeper_X_Axis+20,480),CV_RGB(255,0,0),12,8);
(Coordinate_X << GoalKeeper_X_Axis);
(Coordinate_Y << 480);
Coordinate_Final << "(" << Coordinate_X.str() << "," << Coordinate_Y.str() << ")";
putText(GoalKeeper_Window,Coordinate_Final.str(),Point((GoalKeeper_X_Axis),(480-10)),5,1,CV_RGB(255,0,0));
imshow(GoalKeeper_Wndow,drawing);
char q = waitKey(2);
if(q == 'q')
{
Trajectory_Window.setTo(Scalar(0,0,0));
}
/* Clear the window */
GoalKeeper_Window.setTo(Scalar(0,0,0));
Coordinate_Final.str(std::string());
Coordinate_X.str(std::string());
Coordinate_Y.str(std::string());
#ifdef TIME_GOALKEEPER
GoalKeeper_Count++;
if(GoalKeeper_Count == 700)
{
clock_gettime(CLOCK_REALTIME, &GoalKeeper_Stop_Time);
printf("[GOAL_KEEPER_MOVEMENT] Count: %d Stop Time: Sec= %ld and Nsec= %ld \n",GoalKeeper_Count, GoalKeeper_Stop_Time.tv_sec, GoalKeeper_Stop_Time.tv_nsec);
diff_time(&GoalKeeper_Stop_Time, &GoalKeeper_Start_Time, &GoalKeeper_Diff);
printf("[GOAL_KEEPER_MOVEMENT] Time Difference: Sec= %ld and Nsec= %ld \n", GoalKeeper_Diff.tv_sec, GoalKeeper_Diff.tv_nsec);
exit(0);
}
#endif
}
}
/******************************************************************//****
* @brief Video_Output()
* This function stores the images captured as a Video
*
***********************************************************************/
void *Video_Output(void * param)
{
printf("[VIDEO_OUTPUT] Started Task\n");
VideoWriter wrt;
Mat retrieve_element;
string namemove("Kovuru.avi");
/* Opening a video writer */
cv::VideoWriter output_cap(namemove,CV_FOURCC('M','J','P','G'), 1, cv::Size ( 640,480), true);
/* Check whether the video writer is created or not */
if (!output_cap.isOpened())
{
std::cout << "Video Can't be opened" << std::endl;
pthread_exit(NULL);
}
while(1)
{
#ifdef TIME_VIDEOOUTPUT
if(VideoOutput_Count == 0)
{
clock_gettime(CLOCK_REALTIME, &VideoOuput_Start_Time);
printf("[VIDEO_OUTPUT] Count: %d Start Time: Sec= %ld and Nsec= %ld \n",VideoOutput_Count, VideoOuput_Start_Time.tv_sec, VideoOuput_Start_Time.tv_nsec);
}
#endif
/* Store frames as window */
pthread_mutex_lock(&VideoOutput_Mutex);
if(VideoOutput_Count != 0)
{
Images[VideoOutput_Count].copyTo(retrieve_element);
output_cap.write(retrieve_element);
VideoOutput_Count--;
}
pthread_mutex_unlock(&VideoOutput_Mutex);
#ifdef TIME_VIDEOOUTPUT
VideoOutput_Count++;
if(VideoOutput_Count == 700)
{
clock_gettime(CLOCK_REALTIME, &VideoOuput_Stop_Time);
printf("[VIDEO_OUTPUT] Count: %d Stop Time: Sec= %ld and Nsec= %ld \n",VideoOutput_Count, GoalKeeper_Stop_Time.tv_sec, GoalKeeper_Stop_Time.tv_nsec);
diff_time(&VideoOuput_Stop_Time, &VideoOuput_Start_Time, &VideoOuput_Diff);
printf("[VIDEO_OUTPUT] Time Difference: Sec= %ld and Nsec= %ld \n", VideoOuput_Diff.tv_sec, VideoOuput_Diff.tv_nsec);
exit(0);
}
#endif
}
/* Release the video writer */
output_cap.release();
}
/******************************************************************//****
* @brief Sequencer_Thread()
* This function deploys interdependent threads sequentially
*
***********************************************************************/
void *Sequencer_Thread(void * param)
{
while(1)
{
usleep(120000);
sem_post(&Coordinate_Sem);
usleep(70000);
sem_post(&GoalKeeper_Sem);
usleep(55000);
sem_post(&Coordinate_Sem);
usleep(120000);
sem_post(&Coordinate_Sem);
usleep(20000);
sem_post(&GoalKeeper_Sem);
usleep(120000);
sem_post(&Coordinate_Sem);
usleep(120000);
sem_post(&GoalKeeper_Sem);
usleep(20000);
sem_post(&Coordinate_Sem);
usleep(120000);
sem_post(&Coordinate_Sem);
usleep(55000);
sem_post(&GoalKeeper_Sem);
usleep(70000);
sem_post(&Coordinate_Sem);
usleep(120000);
sem_post(&Coordinate_Sem);sem_post(&GoalKeeper_Sem);
}
}
/******************************************************************//****
* @brief main()
* This function creates different threads and semaphores required.
* And, Finaaly releases the resorces.
*
***********************************************************************/
int main (int argc, char *argv[])
{
int max_prio, min_prio;
cpu_set_t cpu1,cpu2,cpu3;
/******** Creating output windows to show the results *******/
/* Create different windows for goal keeper and centroid */
namedWindow( GoalKeeper_Wndow, CV_WINDOW_AUTOSIZE );
namedWindow( ObjectDetetion_Window, CV_WINDOW_AUTOSIZE );
/* capture image from camera. dev=0 or 1 depending on the device */
cap = (CvCapture *)cvCreateCameraCapture(dev);
/******** Setting resolution ********************/
cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, 640);
cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, 480);
/* Printing the schedular policy */
printf("[MAIN] Initial Scheduling Policy\n");
Display_Scheduler();
/* Initializaion of CPU set for different threads*/
CPU_ZERO(&cpu1);
CPU_SET(0, &cpu1);
CPU_ZERO(&cpu2);
CPU_SET(1, &cpu2);
CPU_ZERO(&cpu3);
CPU_SET(2, &cpu3);
/******* Semaphore Creation *****/
sem_init(&Coordinate_Sem, 0, 1);
sem_init(&GoalKeeper_Sem, 0, 1)
/***** Mutex Creation *********/
/* Set default protocol for mutex */
pthread_mutex_init(&FrameCapture_Mutex, NULL);
pthread_mutex_init(&Coordinate_Mutex, NULL);
pthread_mutex_init(&VideoOutput_Mutex, NULL);
/******** Initialize Task Attributes ********/
/* Assigning the attributes for the threads */
pthread_attr_init (&Frame_Sched_Attr);
pthread_attr_init (&ObjectDetect_Sched_Attr);
pthread_attr_init (&GoalKeeper_Sched_Attr);
pthread_attr_init (&Sequencer_Shed_Attr);
pthread_attr_init (&VideoOutput_Sched_Attr);
/* Assign Policies */
pthread_attr_setschedpolicy (&Frame_Sched_Attr, SCHED_FIFO);
pthread_attr_setschedpolicy (&ObjectDetect_Sched_Attr, SCHED_FIFO);
pthread_attr_setschedpolicy (&GoalKeeper_Sched_Attr, SCHED_FIFO);
pthread_attr_setschedpolicy (&Sequencer_Shed_Attr, SCHED_FIFO);
pthread_attr_setschedpolicy (&VideoOutput_Sched_Attr, SCHED_FIFO);
/* Get Maximum and Minimum Priorities */
max_prio = sched_get_priority_max (SCHED_FIFO);
min_prio = sched_get_priority_min (SCHED_FIFO);
/* Assigning priority to threads */
sched_getparam (getpid(), &Main_param);
Main_param.sched_priority = max_prio;
Frame_param.sched_priority = max_prio-20;
ObjectDetection_param.sched_priority = max_prio-30;
GoalKeeper_param.sched_priority = max_prio-40;
Sequencer_param.sched_priority = max_prio-10;
VideoOutput_param.sched_priority = max_prio-50;
/* Assigning main thread with scheduling policy FIFO */
sched_setscheduler(getpid(), SCHED_FIFO, &Main_param);
printf("[MAIN] Scheduling Policy After Modification\n");
Display_Scheduler();
/* assign scheduling parameters */
pthread_attr_setschedparam (&Frame_Sched_Attr, &Frame_param);
pthread_attr_setschedparam (&ObjectDetect_Sched_Attr, &ObjectDetection_param);
pthread_attr_setschedparam (&GoalKeeper_Sched_Attr, &GoalKeeper_param);
pthread_attr_setschedparam (&Sequencer_Shed_Attr, &Sequencer_param);
pthread_attr_setschedparam (&VideoOutput_Sched_Attr, &VideoOutput_param);
/************* Creation of threads ************************/
printf("[MAIN] Creating Threads\n");
pthread_create (&Sequencer_Thread , &Sequencer_Shed_Attr , Sequencer_Thread , NULL );
pthread_create (&Frame_Thread , &Frame_Sched_Attr , Frame_Capture , NULL );
pthread_create (&ObjectDetect_Thread , &ObjectDetect_Sched_Attr , Object_Detection , NULL );
pthread_create (&GoalKeeper_Thread , &GoalKeeper_Sched_Attr , GoalKeeper_Movement , NULL );
pthread_create (&VideoOutput_Thread , &VideoOutput_Sched_Attr , Video_Output , NULL );
/************* Thread Joining ************************/
printf("[MAIN] Waiting On Joining Threads\n");
pthread_join ( Frame_Thread , NULL );
pthread_join ( ObjectDetect_Thread , NULL );
pthread_join ( GoalKeeper_Thread , NULL );
pthread_join ( Sequencer_Thread , NULL );
/* Stop capturing and free the resources */
cvReleaseCapture(&cap);
printf("[MAIN] Released Capture\n");
/****** Destroy Semaphores and Mutexes ******/
sem_destroy(&Coordinate_Sem);
sem_destroy(&GoalKeeper_Sem);
pthread_mutex_destroy(&FrameCapture_Mutex);
pthread_mutex_destroy(&VideoOutput_Mutex);
pthread_mutex_destroy(&Coordinate_Mutex);
printf("[MAIN] [SUCCESS] TEST COMPLETED SUCCESSFULY");
}