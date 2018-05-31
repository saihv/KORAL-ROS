#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "koralROS/CUDAK2NN.h"
#include "koralROS/Keypoint.h"
#include <chrono>

#define cudaCalloc(A, B, STREAM) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemsetAsync(*A, 0, B, STREAM); \
} while (0)

class FeatureMatcher {
public:
	uint64_t *d_descQ, *d_descT;
	uint maxkpNum;	
    std::vector<cv::DMatch> dmatches;

private:
	unsigned int kpTrain, kpQuery;
	unsigned int matchThreshold;
	struct Match {
		int q, t;
		Match() {}
		Match(const int _q, const int _t) : q(_q), t(_t) {}
	};

	struct cudaResourceDesc resDesc;
	struct cudaTextureDesc texDesc;
	cudaTextureObject_t tex_q = 0;
	int* d_matches;
	cudaStream_t m_stream1, m_stream2;

	int* h_matches;

public:
	FeatureMatcher(const uint _thresh, const uint _kpNum) : matchThreshold(_thresh), maxkpNum(_kpNum)
	{
		if (cudaStreamCreate(&m_stream1) == cudaErrorInvalidValue || cudaStreamCreate(&m_stream2) == cudaErrorInvalidValue)
			std::cerr << "Unable to create stream" << std::endl;
		
		cudaCalloc((void**) &d_descQ, 64 * maxkpNum, m_stream1);
		cudaCalloc((void**) &d_descT, 64 * maxkpNum, m_stream2);

		memset(&resDesc, 0, sizeof(resDesc));
		memset(&texDesc, 0, sizeof(texDesc));

		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		resDesc.res.linear.desc.x = 32;
		resDesc.res.linear.desc.y = 32;

		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		cudaMalloc(&d_matches, 4 * maxkpNum);
		h_matches = reinterpret_cast<int*>(malloc(4 * kpQuery));
	}

	~FeatureMatcher()
	{
	}

	void freeGPUMemory()
	{
		cudaFree(d_descQ);
		cudaFree(d_descT);
		cudaFree(d_matches);
	}

	// Allocate memory and transfer descriptors for training image
	void setTrainingImage(std::vector<Keypoint> kps, std::vector<uint64_t> desc)
	{
		kpTrain = kps.size();
		cudaMemsetAsync(d_descT, 0, 64 * (kpTrain + 8), m_stream1);
		cudaMemcpyAsync(d_descT, &desc[0], 64 * (kpTrain + 8), cudaMemcpyHostToDevice, m_stream1);
		cudaStreamSynchronize(m_stream1);
	}

	// Allocate memory and transfer descriptors for query image
	void setQueryImage(std::vector<Keypoint> kps, std::vector<uint64_t> desc)
	{
		kpQuery = kps.size();
		cudaMemsetAsync(d_descQ, 0, 64 * (kpQuery), m_stream2);
		cudaMemcpyAsync(d_descQ, &desc[0], 64 * (kpQuery), cudaMemcpyHostToDevice, m_stream2);
		cudaStreamSynchronize(m_stream2);

		resDesc.res.linear.devPtr = d_descQ;
		resDesc.res.linear.sizeInBytes = 64 * kps.size();

		cudaCreateTextureObject(&tex_q, &resDesc, &texDesc, nullptr);
	}

	// Perform brute force matching between training and query images
	void matchFeatures()
	{
		cudaMemset(d_matches, 0, static_cast<int>(kpQuery));
		auto start = high_resolution_clock::now();	
		CUDAK2NN(d_descT, static_cast<int>(kpTrain), tex_q, static_cast<int>(kpQuery), d_matches, matchThreshold);
		auto end = high_resolution_clock::now();

		std::vector<int> h_matches(kpQuery);
		cudaMemcpy(&h_matches[0], d_matches, 4 * kpQuery, cudaMemcpyDeviceToHost);
		std::vector<Match> matches;
		dmatches.clear();
		for (size_t i = 0; i < kpQuery; ++i) {
			if (h_matches[i] != -1) {
				matches.emplace_back(i, h_matches[i]);
				dmatches.emplace_back(h_matches[i], i, 0.0f);
			}
		}
		
		auto sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(1);
		std::cout << "Computed " << matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;		
	}
};