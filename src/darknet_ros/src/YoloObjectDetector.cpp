/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"
#include "std_msgs/Float64MultiArray.h"
using namespace std;
using namespace cv;
// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros
{

char *cfg;

char *weights;

char *data;

char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
	: nodeHandle_(nh), imageTransport_(nodeHandle_), numClasses_(0), classLabels_(0), rosBoxes_(0), rosBoxCounter_(0)
{
	ROS_INFO("[YoloObjectDetector] Node started.");
	// Read parameters from config file.
	if (!readParameters()) {
		ros::requestShutdown();
	}
	bottle_img = cv::Mat(cv::Size(100, 100), CV_8UC3, cv::Scalar(0, 0, 0));
	bottleCnt = 0;
//	new_bottles = false;
	init(nh);
}

YoloObjectDetector::~YoloObjectDetector()
{
	{
		boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
		isNodeRunning_ = false;
	}
	yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
	// Load common parameters.
	nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
	nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
	nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

	// Check if Xserver is running on Linux.
	if (XOpenDisplay(NULL)) {
		// Do nothing!
		ROS_INFO("[YoloObjectDetector] Xserver is running.");
	}
	else {
		ROS_INFO("[YoloObjectDetector] Xserver is not running.");
		viewImage_ = false;
	}

	// Set vector sizes.
	nodeHandle_.param("yolo_model/detection_classes/names", classLabels_, std::vector<std::string>(0));
	numClasses_ = classLabels_.size();
	rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
	rosBoxCounter_ = std::vector<int>(numClasses_);

	return true;
}

void YoloObjectDetector::init(ros::NodeHandle& nh)
{
	ROS_INFO("[YoloObjectDetector] init().");

	// Initialize deep network of darknet.
	std::string weightsPath;
	std::string configPath;
	std::string dataPath;
	std::string configModel;
	std::string weightsModel;

	// Threshold of object detection.
	float thresh;
	nodeHandle_.param("yolo_model/threshold/value", thresh, (float)0.3);

	// Path to weights file.
	nodeHandle_.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
	nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
	weightsPath += "/" + weightsModel;
	weights = new char[weightsPath.length() + 1];
	strcpy(weights, weightsPath.c_str());

	// Path to config file.
	nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));  //默认yolo2 tiny
	nodeHandle_.param("config_path", configPath, std::string("/default"));
	configPath += "/" + configModel;
	cfg = new char[configPath.length() + 1];
	strcpy(cfg, configPath.c_str());

	// Path to data folder.
	dataPath = darknetFilePath_;
	dataPath += "/data";
	data = new char[dataPath.length() + 1];
	strcpy(data, dataPath.c_str());

	// Get classes.
	detectionNames = (char **)realloc((void *)detectionNames, (numClasses_ + 1) * sizeof(char *));
	for (int i = 0; i < numClasses_; i++) {
		detectionNames[i] = new char[classLabels_[i].length() + 1];
		strcpy(detectionNames[i], classLabels_[i].c_str());
	}

	// Load network.
	setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_, 0, 0, 1, 0.5, 0, 0, 0, 0);
	yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

    image_transport::ImageTransport it(nh);  //image_transport
    bottle_img_pub = it.advertise("/darknet/bottle_img", 100);
    bottle_box_pub = nh.advertise<std_msgs::Float64MultiArray>("/darknet/bottle_box", 10);

	// Initialize publisher and subscriber.
	std::string cameraTopicName;
	int cameraQueueSize;
	std::string objectDetectorTopicName;
	int objectDetectorQueueSize;
	bool objectDetectorLatch;
	std::string boundingBoxesTopicName;
	int boundingBoxesQueueSize;
	bool boundingBoxesLatch;
	std::string detectionImageTopicName;
	int detectionImageQueueSize;
	bool detectionImageLatch;

	nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image_raw"));
	nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
	nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName, std::string("found_object"));
	nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
	nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
	nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName, std::string("bounding_boxes"));
	nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
	nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
	nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName, std::string("detection_image"));
	nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
	nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

//    imageSubscriber_ = //传递参数进行压缩图像的订阅
//        nh.subscribe(cameraTopicName, cameraQueueSize, &YoloObjectDetector::cameraCallback_compressed, this);


    imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &YoloObjectDetector::cameraCallback, this);

	objectPublisher_ =
		nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>(objectDetectorTopicName,
															 objectDetectorQueueSize,
															 objectDetectorLatch);
	boundingBoxesPublisher_ =
		nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(boundingBoxesTopicName,
															   boundingBoxesQueueSize,
															   boundingBoxesLatch);
	detectionImagePublisher_ =
		nodeHandle_
			.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);

	// Action servers.
	std::string checkForObjectsActionName;
	nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName, std::string("check_for_objects"));
	checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
	checkForObjectsActionServer_
		->registerGoalCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
	checkForObjectsActionServer_
		->registerPreemptCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
	checkForObjectsActionServer_->start();

}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &msg)
{
	ROS_DEBUG("[YoloObjectDetector] USB image received.");

	cv_bridge::CvImagePtr cam_image;

	try {
		cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	if (cam_image) {
		{
			boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
			imageHeader_ = msg->header;
			camImageCopy_ = cam_image->image.clone();
		}
		{
			boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
			imageStatus_ = true;
		}
		frameWidth_ = cam_image->image.size().width;
		frameHeight_ = cam_image->image.size().height;
	}
	return;
}

void YoloObjectDetector::cameraCallback_compressed(const sensor_msgs::CompressedImageConstPtr &msg_img_left)
{
    ROS_DEBUG("[YoloObjectDetector] USB image received.");

    cv_bridge::CvImagePtr cam_image;

    try {
        cam_image = cv_bridge::toCvCopy(msg_img_left, sensor_msgs::image_encodings::BGR8);
//        cam_image = cv_ptr_img_left->image;
//        cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (cam_image) {
        {
            boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
            imageHeader_ = msg_img_left->header;
            camImageCopy_ = cam_image->image.clone();
        }
        {
            boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
            imageStatus_ = true;
        }
        frameWidth_ = cam_image->image.size().width;
        frameHeight_ = cam_image->image.size().height;
    }
    return;
}


void YoloObjectDetector::checkForObjectsActionGoalCB()
{
	ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

	boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal>
		imageActionPtr = checkForObjectsActionServer_->acceptNewGoal();
	sensor_msgs::Image imageAction = imageActionPtr->image;

	cv_bridge::CvImagePtr cam_image;

	try {
		cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	if (cam_image) {
		{
			boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
			camImageCopy_ = cam_image->image.clone();
		}
		{
			boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
			actionId_ = imageActionPtr->id;
		}
		{
			boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
			imageStatus_ = true;
		}
		frameWidth_ = cam_image->image.size().width;
		frameHeight_ = cam_image->image.size().height;
	}
	return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
	ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
	checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
	return (ros::ok() && checkForObjectsActionServer_->isActive()
		&& !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat &detectionImage)
{
	if (detectionImagePublisher_.getNumSubscribers() < 1) return false;
	cv_bridge::CvImage cvImage;
	cvImage.header.stamp = ros::Time::now();
	cvImage.header.frame_id = "detection_image";
	cvImage.encoding = sensor_msgs::image_encodings::BGR8;
	cvImage.image = detectionImage;
	detectionImagePublisher_.publish(*cvImage.toImageMsg());
	ROS_DEBUG("Detection image has been published.");
	return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
	int i;
	int count = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			count += l.outputs;
		}
	}
	return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
	int i;
	int count = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
	int i, j;
	int count = 0;
	fill_cpu(demoTotal_, 0, avg_, 1);
	for (j = 0; j < demoFrame_; ++j) {
		axpy_cpu(demoTotal_, 1. / demoFrame_, predictions_[j], 1, avg_, 1);
	}
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
	detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
	return dets;
}

void *YoloObjectDetector::detectInThread()
{
	running_ = 1;
	float nms = .4;

	layer l = net_->layers[net_->n - 1];
	float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
	float *prediction = network_predict(net_, X);
	rememberNetwork(net_);
	detection *dets = 0;
	int nboxes = 0;
	dets = avgPredictions(net_, &nboxes);

	if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

	if (enableConsoleOutput_) {

//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.1f\n", fps_);
//    printf("Objects:\n\n");
	}

	image display = buff_[(buffIndex_ + 2) % 3];

	draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);
	// extract the bounding boxes and send them to ROS
	int i, j;
	int count = 0;
	for (i = 0; i < nboxes; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;
		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > 1) xmax = 1;
		if (ymax > 1) ymax = 1;

		// iterate through possible boxes and collect the bounding boxes
//		std::cout << "**********  frame result *************" << std::endl;

		for (j = 0; j < demoClasses_; ++j) {
			if (dets[i].prob[j]) {
				float x_center = (xmin + xmax) / 2;
				float y_center = (ymin + ymax) / 2;
				float BoundingBox_width = xmax - xmin;
				float BoundingBox_height = ymax - ymin;

				// define bounding box
				// BoundingBox must be 1% size of frame (3.2x2.4 pixels)
				if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
					roiBoxes_[count].x = x_center;
					roiBoxes_[count].y = y_center;
					roiBoxes_[count].w = BoundingBox_width;
					roiBoxes_[count].h = BoundingBox_height;
					roiBoxes_[count].Class = j;
					roiBoxes_[count].prob = dets[i].prob[j];
					roiBoxes_[count].name = demoNames_[j];
//todo  show detect info

//					std::cout << "This " << roiBoxes_[count].prob * 100 << "% possibilities is a "
//							  << roiBoxes_[count].name << std::endl;
//					std::cout << "    center position: <" << roiBoxes_[count].x * display.w << ","
//							  << roiBoxes_[count].y * display.h << ">" << std::endl;
//					std::cout << "           box size: <" << roiBoxes_[count].w * display.w << ","
//							  << roiBoxes_[count].h * display.h << ">" << std::endl;
//					std::cout << "--------------------------------" << std::endl;

					count++;
				}
			}
		}
	}

	// create array to store found bounding boxes
	// if no object detected, make sure that ROS knows that num = 0
	if (count == 0) {
		roiBoxes_[0].num = 0;
	}
	else {
		roiBoxes_[0].num = count;
	}

	free_detections(dets, nboxes);
	demoIndex_ = (demoIndex_ + 1) % demoFrame_;
	running_ = 0;
	return 0;
}

void *YoloObjectDetector::fetchInThread()
{
	{
		boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
		IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();  //获取到新来的帧
		IplImage *ROS_img = imageAndHeader.image;
		ipl_into_image(ROS_img, buff_[buffIndex_]);
		headerBuff_[buffIndex_] = imageAndHeader.header;
		buffId_[buffIndex_] = actionId_;
	}
	rgbgr_image(buff_[buffIndex_]);
	letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
	return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{

//	cv::Mat Img;
//	Img = cv::cvarrToMat(ipl_);
//	cv::imshow("test",Img);

	show_image_cv(buff_[(buffIndex_ + 1) % 3], "YOLO V3", ipl_);

//	cv::Mat Img;
//	Img = cv::cvarrToMat(ipl_);
//	cv::imshow("test",Img);

	int c = cv::waitKey(waitKeyDelay_);
	if (c != -1) c = c % 256;
	if (c == 27) {
		demoDone_ = 1;
		return 0;
	}
	else if (c == 82) {
		demoThresh_ += .02;
	}
	else if (c == 84) {
		demoThresh_ -= .02;
		if (demoThresh_ <= .02) demoThresh_ = .02;
	}
	else if (c == 83) {
		demoHier_ += .02;
	}
	else if (c == 81) {
		demoHier_ -= .02;
		if (demoHier_ <= .0) demoHier_ = .0;
	}
	return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
	while (1) {
		displayInThread(0);
	}
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
	while (1) {
		detectInThread();
	}
}

void YoloObjectDetector::setupNetwork(char *cfgfile,
									  char *weightfile,
									  char *datafile,
									  float thresh,
									  char **names,
									  int classes,
									  int delay,
									  char *prefix,
									  int avg_frames,
									  float hier,
									  int w,
									  int h,
									  int frames,
									  int fullscreen)
{
	demoPrefix_ = prefix;
	demoDelay_ = delay;
	demoFrame_ = avg_frames;
	image **alphabet = load_alphabet_with_file(datafile);
	demoNames_ = names;
	demoAlphabet_ = alphabet;
	demoClasses_ = classes;
	demoThresh_ = thresh;
	demoHier_ = hier;
	fullScreen_ = fullscreen;
	printf("YOLO V3\n");
	net_ = load_network(cfgfile, weightfile, 0);
	set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{  // main process loop
	const auto wait_duration = std::chrono::milliseconds(2000);
	while (!getImageStatus()) {
		printf("Waiting for image.\n");
		if (!isNodeRunning()) {
			return;
		}
		std::this_thread::sleep_for(wait_duration);
	}

	std::thread detect_thread;
	std::thread fetch_thread;

	srand(2222222);

	int i;
	demoTotal_ = sizeNetwork(net_);
	predictions_ = (float **)calloc(demoFrame_, sizeof(float *));
	for (i = 0; i < demoFrame_; ++i) {
		predictions_[i] = (float *)calloc(demoTotal_, sizeof(float));
	}
	avg_ = (float *)calloc(demoTotal_, sizeof(float));

	layer l = net_->layers[net_->n - 1];
	roiBoxes_ = (darknet_ros::RosBox_ *)calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

	{
		boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
		IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
		IplImage *ROS_img = imageAndHeader.image;
		buff_[0] = ipl_to_image(ROS_img);
		headerBuff_[0] = imageAndHeader.header;
	}
	buff_[1] = copy_image(buff_[0]);
	buff_[2] = copy_image(buff_[0]);
	headerBuff_[1] = headerBuff_[0];
	headerBuff_[2] = headerBuff_[0];
	buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
	buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
	buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
	ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

	int count = 0;

	if (!demoPrefix_ && viewImage_) {
		if (fullScreen_) {
            if(show_imgs)
            {
                cv::namedWindow("YOLO V3", cv::WINDOW_NORMAL);
                cv::setWindowProperty("YOLO V3", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            }
		}
		else {
			//原先
//			cv::moveWindow("YOLO V3", 0, 0);
//			cv::resizeWindow("YOLO V3", 640, 480);

            //改编的，不让显示
            if(show_imgs){
                cv::namedWindow("YOLO V3 Bottle", CV_WINDOW_AUTOSIZE);
                cv::namedWindow("YOLO V3", CV_WINDOW_AUTOSIZE);
            }
		}
	}

	demoTime_ = what_time_is_it_now();

	while (!demoDone_) {  //main processing loop-----------------------------
		buffIndex_ = (buffIndex_ + 1) % 3;
		fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
		detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);

		if (!demoPrefix_) {


			fps_ = 1. / (what_time_is_it_now() - demoTime_);
			demoTime_ = what_time_is_it_now();
			if (viewImage_) {
//				这里就不显示所有检测出来的东西了，只显示bottle
//				displayInThread(0);

			}
			else {
				generate_image(buff_[(buffIndex_ + 1) % 3], ipl_);
			}
			publishInThread();
			//显示bottle 处理bottle
			if (Bottle_rects.size() != 0) {
				processBottle();
			}

            if(show_imgs)
            {
                cv::imshow("YOLO V3 Bottle", bottle_img);
                cv::imshow("YOLO V3", camImageCopy_dev);
                cv::waitKey(1);
            }

		}
		else {
			char name[256];
			sprintf(name, "%s_%08d", demoPrefix_, count);
			save_image(buff_[(buffIndex_ + 1) % 3], name);
		}
		fetch_thread.join();
		detect_thread.join();
		++count;
		if (!isNodeRunning()) {
			demoDone_ = true;
		}
	}
}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
	IplImage *ROS_img = new IplImage(camImageCopy_);
	camImageCopy_.copyTo(camImageCopy_dev);
	IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
	return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
	boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
	return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
	boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
	return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
	// Publish image.
	cv::Mat cvImage = cv::cvarrToMat(ipl_);
	if (!publishDetectionImage(cv::Mat(cvImage))) {
		ROS_DEBUG("Detection image has not been broadcasted.");
	}

	// Publish bounding boxes and detection result.
	int num = roiBoxes_[0].num;  // 0 处存储着本帧的目标检测信息
	if (num > 0 && num <= 100) {
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < numClasses_; j++) {
				if (roiBoxes_[i].Class == j) {
					rosBoxes_[j].push_back(roiBoxes_[i]);
					rosBoxCounter_[j]++;
				}
			}
		}

		darknet_ros_msgs::ObjectCount msg;
		msg.header.stamp = ros::Time::now();
		msg.header.frame_id = "detection";
		msg.count = num;
		objectPublisher_.publish(msg);
		Bottle_rects.clear();
		for (int i = 0; i < numClasses_; i++) {
			if (rosBoxCounter_[i] > 0) {
				darknet_ros_msgs::BoundingBox boundingBox;

				for (int j = 0; j < rosBoxCounter_[i]; j++) {
					int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
					int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
					int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
					int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

					std::cout << "This " << rosBoxes_[i][j].prob * 100 << "% possibilities is a "
							  << rosBoxes_[i][j].name << std::endl;
					std::cout << "       top left: <" << xmin << "," << ymin << ">" << std::endl;
					std::cout << "   bottmn right: <" << xmax << "," << ymax << ">" << std::endl;
					std::cout << "--------------------------------" << std::endl;

					if (rosBoxes_[i][j].name == "bottle") {
						cv::Rect bottle_roi = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

						Bottle_rects.emplace_back(std::make_pair(rosBoxes_[i][j].prob, bottle_roi));
					}
					boundingBox.Class = classLabels_[i];
					boundingBox.id = i;
					boundingBox.probability = rosBoxes_[i][j].prob;
					boundingBox.xmin = xmin;
					boundingBox.ymin = ymin;
					boundingBox.xmax = xmax;
					boundingBox.ymax = ymax;
					boundingBoxesResults_.bounding_boxes.push_back(boundingBox);

				}
			}
		}
		boundingBoxesResults_.header.stamp = ros::Time::now();
		boundingBoxesResults_.header.frame_id = "detection";
		boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
		boundingBoxesPublisher_.publish(boundingBoxesResults_);
		std::cout << num << "detection published" << std::endl;
	}
	else {
		darknet_ros_msgs::ObjectCount msg;
		msg.header.stamp = ros::Time::now();
		msg.header.frame_id = "detection";
		msg.count = 0;
		objectPublisher_.publish(msg);
	}
	if (isCheckingForObjects()) {
		ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
		darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
		objectsActionResult.id = buffId_[0];
		objectsActionResult.bounding_boxes = boundingBoxesResults_;
		checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
	}
	boundingBoxesResults_.bounding_boxes.clear();
	for (int i = 0; i < numClasses_; i++) {
		rosBoxes_[i].clear();
		rosBoxCounter_[i] = 0;
	}

	return 0;
}

void YoloObjectDetector::processBottle(){
//	todo 绘图
	cv::Mat image = camImageCopy_.clone();
//	camImageCopy_dev.copyTo(bottle_img);
	bottle_img = cv::Mat(camImageCopy_dev.size(), camImageCopy_dev.type(), cv::Scalar(0, 0, 0));

	for (auto &it:Bottle_rects) {

//		if (it.first < 0.6f) {
//			continue;
//		}

//		在图像上将水瓶框选出来

		//todo 发布消息
        std_msgs::Float64MultiArray box_msg;
        box_msg.data = std::vector<double>{(double) it.second.x, (double) it.second.y,
                                           (double) it.second.width, (double) it.second.height,
                                           it.first*100.0f
        };

        bottle_box_pub.publish(box_msg);
        sensor_msgs::ImagePtr sub_msg;
        std_msgs::Header header;
        header.stamp = ros::Time::now();

        sub_msg = cv_bridge::CvImage(header, "bgr8", camImageCopy_dev(it.second)).toImageMsg();
        bottle_img_pub.publish(sub_msg);

        cv::rectangle(camImageCopy_dev, it.second, cv::Scalar(0, 0, 255), 3);

//        bottle_img_pub
//            bottle_box_pub

		std::string prb = "prob: " + std::to_string(it.first * 100.0f).substr(0, 5) + "%";
		int x = it.second.x;
		int y = it.second.y;
		int offset = 5;
		y -= offset;

//		防止超出图像
		get_resize(camImageCopy_dev.cols, camImageCopy_dev.rows, x, y);
		cv::Point text_pos(x, y);
		cv::putText(image, prb, text_pos, 1, 1.5, cv::Scalar(0, 0, 255), 3);

		//----------- devid images
		x = it.second.x;
		y = it.second.y;

		cv::Mat bottle_sub;
		int w = it.second.width;
		int h = it.second.height;
//		拓宽瓶子的区域，长宽各自增加1/3的大小
		cv::Point2d expand(w / 6, h / 6);
		cv::Point2i left_top(x - expand.x, y - expand.y);
		cv::Point2i right_bott(x + w + expand.x, y + h + expand.y);

//		防止超出图像
		get_resize(camImageCopy_dev.cols, camImageCopy_dev.rows, left_top.x, left_top.y);
		get_resize(camImageCopy_dev.cols, camImageCopy_dev.rows, right_bott.x, right_bott.y);

		cv::Rect new_rect = cv::Rect(left_top, right_bott);
		cv::rectangle(camImageCopy_dev, new_rect, cv::Scalar(0, 255, 0), 1);

		//进行瓶盖检测，如果检测不到瓶盖的话那么说明就不是一个我们需要的水瓶
		Mat TargetBottle;
		camImageCopy_(new_rect).copyTo(TargetBottle);
//		if (!detectBottleCover(TargetBottle)) continue;  //jetcam或者其它彩色图像的时候使用

//		检测到我们所需要的水瓶，那么就将对应像素换成水瓶
		TargetBottle.copyTo(bottle_img(new_rect));
		if (cv::waitKey(1) == 's') {  //如果按下按键‘s’ 那么保存该次检测到的水瓶
			cv::imwrite("/home/lab/Project/RobotPrj/target_detect/deep_learning_method/output/mynteye/water"
							+ std::to_string(bottleCnt) + ".jpg",
						bottle_img(new_rect));
		}

		cv::putText(bottle_img, prb, left_top, 1, 1.5, cv::Scalar(0, 0, 255), 3);

//		bottle_img(new_rect) = image(new_rect).clone();
		bottleCnt++;
	}
	Bottle_rects.clear();
	//todo 将感兴趣区域截图出来
}

bool YoloObjectDetector::detectBottleCover(cv::Mat& coverRegin){

	Scalar_<unsigned char> bgrPixel;
	//假设待检测的瓶盖的颜色rgb的范围
	Scalar_<int> mask = Scalar_<int>(142, 90, 78); //b g r
	int scale_b = 20;
	int scale_g = 30;
	int scale_r = 30;

	double invalid_pixel = 0.0;
	int cnt = 0;

	cv::Mat water_buttle_mask = coverRegin.clone();
	Point2i center(0, 0);

//		圈出一部分一部分瓶盖出现概率较高的检测区域进行检测
	cv::Rect cover_roi = cv::Rect(
		(coverRegin.cols) / 5,
		0,
		(3 * coverRegin.cols) / 5,
		(coverRegin.rows) / 2
	);

	cv::rectangle(water_buttle_mask, cover_roi, cv::Scalar(0, 0, 255), 3);
	cv::Mat cover_img = water_buttle_mask(cover_roi);
//		一张尺寸和原始图像一样的全白的图像，用作标记
	cv::Mat binnary_mask = cv::Mat(cover_img.size(), CV_8UC1, cv::Scalar(255));

	//todo 这一段代码可不可以使用opencv自带的函数代替
	for (int r = 0; r < cover_img.rows; r++) {
		unsigned char *rowPtr = cover_img.row(r).data;
		unsigned char *outputPtr = water_buttle_mask.row(r).data;

//			for (int c = (water_buttle.cols)/3; c < (water_buttle.cols)*(2*3); c++)
		for (int c = 0; c < (cover_img.cols); c++) {
			bgrPixel.val[0] = rowPtr[c * 3 + 0]; // B
			bgrPixel.val[1] = rowPtr[c * 3 + 1]; // G
			bgrPixel.val[2] = rowPtr[c * 3 + 2]; // R

			if (
				abs((int)bgrPixel.val[0] - mask.val[0]) < scale_b &&
					abs((int)bgrPixel.val[1] - mask.val[1]) < scale_g &&
					abs((int)bgrPixel.val[2] - mask.val[2]) < scale_r
				) {

//				如果待该像素点符合瓶盖像素特点的话将对应像素点涂黑
				outputPtr[(c + cover_roi.x) * 3 + 0] = 0;
				outputPtr[(c + cover_roi.x) * 3 + 1] = 0;
				outputPtr[(c + cover_roi.x) * 3 + 2] = 0;
//					如果待该像素点符合瓶盖像素特点的话将mask的对应像素点涂黑
				binnary_mask.at<uchar>(r, c) = (uchar)0;
//				center.x += (c + cover_roi.x);
//				center.y += (r + cover_roi.y);
//				cnt++;
			}
		}
	}

	//非连通域移除 方法一：
//		std::vector<std::vector<cv::Point>> contours;
//		cv::findContours(binnary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
//		contours.erase(std::remove_if(contours.begin(), contours.end(),
//									  [](const std::vector<cv::Point>& c){return cv::contourArea(c) < 200; }), contours.end());
//		binnary_mask.setTo(0);
//		cv::drawContours(binnary_mask, contours, -1, cv::Scalar(255), cv::FILLED);

	//非连通域移除 方法二：
	RemoveSmallRegion(binnary_mask, binnary_mask, 100, 1, 1);
	RemoveSmallRegion(binnary_mask, binnary_mask, 100, 0, 0);
	center = Point2i(0, 0);
	cnt = 0;
	for (int r = 0; r < cover_img.rows; r++) {
		for (int c = 0; c < (cover_img.cols); c++) {
			if (
				binnary_mask.at<uchar>(r, c) == (uchar)0
				) {
				center.x += (c + cover_roi.x);
				center.y += (r + cover_roi.y);
				cnt++;
			}
		}
	}
	double pixel_rate = 100.0 * (double)cnt / (double)(cover_img.rows * cover_img.cols);
//	if (pixel_rate<=0.4 || pixel_rate>= 2 ) return false;
//	if (pixel_rate<=0.1 ) return false;
	if (cnt <= 10) return false;

	cv::putText(coverRegin, to_string(pixel_rate), center/cnt, 1, 1.5, cv::Scalar(0, 0, 255), 3);

//	imshow("binnary_mask", binnary_mask);

	center /= cnt;
	cv::drawMarker(coverRegin, center, cv::Scalar(0, 0, 255), 2,15,2);
	//CV_EXPORTS_W void drawMarker(CV_IN_OUT Mat& img, Point position, const Scalar& color,
	//                             int markerType = MARKER_CROSS, int markerSize=20, int thickness=1,
	//                             int line_type=8);

	return true;
};

void YoloObjectDetector::RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount=0;       //记录除去的个数
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
	cv::Mat Pointlabel = cv::Mat::zeros( Src.size(), CV_8UC1 );

	if(CheckMode==1)
	{
		std::cout<<"Mode: remove outlier. ";
		for(int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for(int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		cout<<"Mode: remove hole. ";
		for(int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for(int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point2i> NeihborPos;  //记录邻域点位置
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode==1)
	{
//		cout<<"Neighbor mode: 8邻域."<<endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
//	else cout<<"Neighbor mode: 4邻域."<<endl;
	int NeihborCount=4+4*NeihborMode;
	int CurrX=0, CurrY=0;
	//开始检测
	for(int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for(int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点
				GrowBuffer.push_back( Point2i(j, i) );
				Pointlabel.at<uchar>(i, j)=1;
				int CheckResult=0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出

				for ( int z=0; z<GrowBuffer.size(); z++ )
				{

					for (int q=0; q<NeihborCount; q++)                                      //检查四个邻域点
					{
						CurrX=GrowBuffer.at(z).x+NeihborPos.at(q).x;
						CurrY=GrowBuffer.at(z).y+NeihborPos.at(q).y;
						if (CurrX>=0&&CurrX<Src.cols&&CurrY>=0&&CurrY<Src.rows)  //防止越界
						{
							if ( Pointlabel.at<uchar>(CurrY, CurrX)==0 )
							{
								GrowBuffer.push_back( Point2i(CurrX, CurrY) );  //邻域点加入buffer
								Pointlabel.at<uchar>(CurrY, CurrX)=1;           //更新邻域点的检查标签，避免重复检查
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult=2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出
				else {CheckResult=1;   RemoveCount++;}
				for (int z=0; z<GrowBuffer.size(); z++)                         //更新Label记录
				{
					CurrX=GrowBuffer.at(z).x;
					CurrY=GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********


			}
		}
	}

	CheckMode=255*(1-CheckMode);
	//开始反转面积过小的区域
	for(int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for(int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if(iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}
//	cout<<RemoveCount<<" objects removed."<<endl;
}

inline void YoloObjectDetector::get_resize(const int &width, const int &height, int &x, int &y)
{
	if (y < 0) y = 0;
	if (y > height) y = height - 1;

	if (x < 0) x = 0;
	if (x > width) x = width - 1;

}

} /* namespace darknet_ros*/
