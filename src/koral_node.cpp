#include "koralROS/Detector.h"
#include "koralROS/Matcher.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class koralROS {
public:
	koralROS(FeatureDetector& detector_, FeatureMatcher& matcher_) : detector(detector_), matcher(matcher_)
	{
		imageL_sub.subscribe(node, "imageL", 1);
		imageR_sub.subscribe(node, "imageR", 1);
		imageSync.reset(new ImageSync(LRSyncPolicy(10), imageL_sub, imageR_sub));
		imageSync->registerCallback(boost::bind(&koralROS::imageCallback, this, _1, _2));
	}

	void imageCallback(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2)
	{
		cv_bridge::CvImagePtr imagePtr1, imagePtr2;
	    imagePtr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::MONO8);
		std::cout << "Left camera: ";
	   	detector.extractFeatures(imagePtr1);
		matcher.setTrainingImage(detector.kps, detector.desc);

		imagePtr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::MONO8);
		std::cout << "Right camera: ";
		detector.extractFeatures(imagePtr2);
		matcher.setQueryImage(detector.kps, detector.desc);
		detector.receivedImg = true;
	}

private:
	ros::NodeHandle node;

	FeatureDetector &detector;
	FeatureMatcher &matcher;
	message_filters::Subscriber <sensor_msgs::Image> imageL_sub, imageR_sub;

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> LRSyncPolicy;
  	
  	typedef message_filters::Synchronizer<LRSyncPolicy> ImageSync;
	boost::shared_ptr<ImageSync> imageSync;
};

int main(int argc, char **argv)
{
  	ros::init(argc, argv, "image_listener");
  	ros::NodeHandle nh;	
  	image_transport::ImageTransport it(nh);

	unsigned int width = 640;
	unsigned int height = 480;
	unsigned int maxFeatureCount = 50000;

	FeatureDetector detector(1.2f, 8, width, height, maxFeatureCount);
	FeatureMatcher matcher(5, maxFeatureCount);
	koralROS koral(detector, matcher);

  	//image_transport::Subscriber sub1 = it.subscribe("imageL", 1, &FeatureDetector::imageCallbackL, &detector);
	//image_transport::Subscriber sub2 = it.subscribe("imageR", 1, &FeatureDetector::imageCallbackR, &detector);


	while(ros::ok()) {
		ros::spinOnce();
		if (detector.receivedImg) {
			matcher.matchFeatures();
			detector.receivedImg = false;
		}
	}
}
