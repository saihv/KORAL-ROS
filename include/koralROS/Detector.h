#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "koralROS/CUDALERP.h"
#include "koralROS/CLATCH.h"
#include "koralROS/FeatureAngle.h"
#include "koralROS/Keypoint.h"
#include "koralROS/KFAST.h"
#include <chrono>

using namespace std::chrono;

class FeatureDetector {
public:
	std::vector<Keypoint> kps;
	std::vector<uint64_t> desc;
	bool receivedImg = false;
	std::vector<cv::KeyPoint> converted_kps;
private:
	struct Level {
		uint8_t* d_img;
		size_t pitch;
		const uint8_t* h_img;
		uint32_t w;
		uint32_t h;
		size_t total;

		Level() : d_img(nullptr), h_img(nullptr) {}
	};

	Level* levels;
	cudaTextureObject_t *all_tex;
	cudaChannelFormatDesc chandesc_img;
	struct cudaTextureDesc texdesc_img;
	cudaTextureObject_t d_trip_tex;
	const float scale_factor;
	const uint8_t scale_levels;
	uint64_t* d_desc;
	Keypoint* d_kps;
	cudaArray* d_img_array;
	cudaTextureObject_t* d_all_tex;
	uint32_t* d_triplets;
	cudaArray* d_trip_arr;
	cudaStream_t* stream = new cudaStream_t[scale_levels - 1];

	const unsigned int width;
	const unsigned int height;
	const unsigned int maxkp;

public:
	FeatureDetector(const float _scale_factor, const uint8_t _scale_levels, const uint _width, const uint _height, const uint _maxkp) : 
	scale_factor(_scale_factor), scale_levels(_scale_levels), width(_width), height(_height), maxkp(_maxkp)
	{
		// Setting cache and shared modes
		cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

		// Allocating and transferring triplets and binding to texture object
		// for CLATCH
		cudaMalloc(&d_triplets, 2048 * sizeof(uint16_t));
		cudaMemcpy(d_triplets, triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);
		cudaChannelFormatDesc chandesc_trip = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&d_trip_arr, &chandesc_trip, 512);
		cudaMemcpyToArray(d_trip_arr, 0, 0, d_triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);
		struct cudaResourceDesc resdesc_trip;
		memset(&resdesc_trip, 0, sizeof(resdesc_trip));
		resdesc_trip.resType = cudaResourceTypeArray;
		resdesc_trip.res.array.array = d_trip_arr;
		struct cudaTextureDesc texdesc_trip;
		memset(&texdesc_trip, 0, sizeof(texdesc_trip));
		texdesc_trip.addressMode[0] = cudaAddressModeClamp;
		texdesc_trip.filterMode = cudaFilterModePoint;
		texdesc_trip.readMode = cudaReadModeElementType;
		texdesc_trip.normalizedCoords = 0;
		cudaCreateTextureObject(&d_trip_tex, &resdesc_trip, &texdesc_trip, nullptr);

		memset(&texdesc_img, 0, sizeof(texdesc_img));
		texdesc_img.addressMode[0] = cudaAddressModeClamp;
		texdesc_img.addressMode[1] = cudaAddressModeClamp;
		texdesc_img.filterMode = cudaFilterModePoint;
		texdesc_img.normalizedCoords = 0;

		chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

		// Setup levels and pre-allocate GPU memory for scale space

		levels = new Level[scale_levels];
		all_tex = new cudaTextureObject_t[scale_levels];
		cudaMalloc(&d_all_tex, scale_levels * sizeof(cudaTextureObject_t));
		float f = 1.0f;
		for (int i = 1; i < scale_levels; ++i) {
			f *= scale_factor;
			levels[i].w = static_cast<uint32_t>(static_cast<float>(width) / f + 0.5f);
			levels[i].h = static_cast<uint32_t>(static_cast<float>(height) / f + 0.5f);
			levels[i].total = static_cast<size_t>(levels[i].w)*static_cast<size_t>(levels[i].h);

			levels[i].h_img = reinterpret_cast<uint8_t*>(malloc(levels[i].total + 1));
			cudaMallocPitch(&levels[i].d_img, &levels[i].pitch, levels[i].w, levels[i].h);

			struct cudaResourceDesc resdesc_img;
			memset(&resdesc_img, 0, sizeof(resdesc_img));
			resdesc_img.resType = cudaResourceTypePitch2D;
			resdesc_img.res.pitch2D.desc = chandesc_img;
			resdesc_img.res.pitch2D.devPtr = levels[i].d_img;
			resdesc_img.res.pitch2D.height = levels[i].h;
			resdesc_img.res.pitch2D.pitchInBytes = levels[i].pitch;
			resdesc_img.res.pitch2D.width = levels[i].w;
			cudaCreateTextureObject(&all_tex[i], &resdesc_img, &texdesc_img, nullptr);
		}

		for (int i = 0; i < scale_levels - 1; ++i) {
			cudaStreamCreate(stream + i);
		}

		// Pre-allocate memory for storing descriptors and keypoint coordinates
		cudaMalloc(&d_desc, 64 * maxkp);
		cudaMalloc(&d_kps, maxkp * sizeof(Keypoint));
		cudaMallocArray(&d_img_array, &chandesc_img, width, height, cudaArrayTextureGather);
	}

	~FeatureDetector()
	{
	}
	
	void imageCallback(const sensor_msgs::ImageConstPtr& msg)
	{
		cv_bridge::CvImagePtr imagePtr;
	        imagePtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
    		extractFeatures(imagePtr);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
   		auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
		receivedImg = true;
	}

	void freeGPUMemory()
	{
		cudaFree(d_desc);
		cudaFree(d_kps);
		cudaFree(d_all_tex);
		for (uint8_t i = 0; i < scale_levels; ++i) {
			cudaFree(levels[i].d_img);
		}

		cudaFree(d_triplets);
		cudaFreeArray(d_trip_arr);
	}

	// Process an image that is read from disk. converted_kps contains keypoints stored in OpenCV format.
	void imageReadProcess(cv::Mat image)
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		detectAndDescribe(image.data, image.cols, image.rows, 60);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		ROS_INFO("Detected %d features in %ld ms \n", kps.size(), duration_cast<milliseconds>( t2 - t1 ).count());
		converted_kps.clear();
		for (const auto& kp : kps) {
			const float scale = static_cast<float>(std::pow(1.2f, kp.scale));
			converted_kps.emplace_back(scale*static_cast<float>(kp.x), scale*static_cast<float>(kp.y), 7.0f*scale, 180.0f / 3.1415926535f * kp.angle, static_cast<float>(kp.score));
		}
	}

	// Process an image that is obtained from a ROS topic. converted_kps contains keypoints stored in OpenCV format.
	void extractFeatures(cv_bridge::CvImagePtr imagePtr)
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		detectAndDescribe(imagePtr->image.data, imagePtr->image.cols, imagePtr->image.rows, 60);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		ROS_INFO("Detected %d features in %ld ms \n", kps.size(), duration_cast<milliseconds>( t2 - t1 ).count());
		converted_kps.clear();
		for (const auto& kp : kps) {
			const float scale = static_cast<float>(std::pow(1.2f, kp.scale));
			converted_kps.emplace_back(scale*static_cast<float>(kp.x), scale*static_cast<float>(kp.y), 7.0f*scale, 180.0f / 3.1415926535f * kp.angle, static_cast<float>(kp.score));
		}
	}	

private:
	void detectAndDescribe(const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t KFAST_thresh)
	{
		// Clear keypoints, assign image and characteristics to the topmost level

		kps.clear();
		levels[0].h_img = image;
		levels[0].w = width;
		levels[0].h = height;
		levels[0].total = static_cast<size_t>(width) * static_cast<size_t>(height);

		// Transfer original image as cudaArray
		// and binding to texture object, one as normalized float (for LERP),
		// one as ElementType (for CLATCH)
		
		cudaTextureObject_t d_img_tex_nf;
		{			
			cudaMemcpyToArray(d_img_array, 0, 0, image, levels[0].total, cudaMemcpyHostToDevice);
			struct cudaResourceDesc resdesc_img;
			memset(&resdesc_img, 0, sizeof(resdesc_img));
			resdesc_img.resType = cudaResourceTypeArray;
			resdesc_img.res.array.array = d_img_array;

			// first as normalized float
			texdesc_img.readMode = cudaReadModeNormalizedFloat;
			cudaCreateTextureObject(&d_img_tex_nf, &resdesc_img, &texdesc_img, nullptr);

			// then as ElementType
			texdesc_img.readMode = cudaReadModeElementType;
			cudaCreateTextureObject(&all_tex[0], &resdesc_img, &texdesc_img, nullptr);
		}
	
		// Prepare the additional scales
		// and bind to ElementType textures
		float f = 1.0f;
		for (int i = 1; i < scale_levels; ++i) {
			f *= scale_factor;			
			cudaMemset2DAsync(levels[i].d_img, levels[i].pitch, 0, levels[i].w, levels[i].h, stream[i - 1]);
			// GPU: non-blocking launch of resize kernels
			CUDALERP(d_img_tex_nf, f, f, levels[i].d_img, levels[i].pitch, levels[i].w, levels[i].h, stream[i - 1]);
		}

		// Initialize KFAST on the CPU

		// Bring in downscale results from GPU (except for first level) 
		// and operate on them as they arrive

		for (uint8_t i = 0; i < scale_levels; ++i) {
			std::vector<Keypoint> local_kps;
			if (i) {
				cudaMemcpy2DAsync(const_cast<uint8_t*>(levels[i].h_img), levels[i].w, levels[i].d_img, levels[i].pitch, levels[i].w, levels[i].h, cudaMemcpyDeviceToHost, stream[i - 1]);
				cudaStreamSynchronize(stream[i - 1]);
			}
			KFAST<true, true>(levels[i].h_img, levels[i].w, levels[i].h, levels[i].w, local_kps, KFAST_thresh);

			// set scale and compute angles
			for (auto& kp : local_kps) {
				kp.scale = i;
				kp.angle = featureAngle(levels[i].h_img, kp.x, kp.y, static_cast<int>(levels[i].w));
			}
			//std::cout << "Got " << local_kps.size() << " keypoints from level " << +i << '.' << std::endl;
			kps.insert(kps.end(), local_kps.begin(), local_kps.end());
		}

		// Compute LATCH descriptors for all the keypoints
		cudaMemcpy(d_all_tex, all_tex, scale_levels * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_kps, kps.data(), kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice);
		CLATCH(d_all_tex, d_trip_tex, d_kps, static_cast<int>(kps.size()), d_desc);

		// Transfer descriptors to host

		desc.clear();
		desc.resize(8 * kps.size());
		cudaMemcpy(&desc[0], d_desc, 64 * kps.size(), cudaMemcpyDeviceToHost);
		
		//cudaDeviceSynchronize();
	}

};
