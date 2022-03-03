#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

#define MINF __int_as_float(0xff800000)
//#define SAVE_IMAGES

void displayBoolean(std::string name, bool *data, int width, int height, bool isGPU, float multiplier = 1.0f) {
	bool *h_data = (bool *)malloc(sizeof(bool) * width * height);

	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(bool) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(bool) * width * height);
	}

	cv::Mat displayMat(height, width, CV_8UC3);
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			if (h_data[h * width + w]) displayMat.at<cv::Vec3b>(h, w) = cv::Vec3b(255, 255, 255);
			else displayMat.at<cv::Vec3b>(h, w) = cv::Vec3b(0, 0, 0);
		}
	}

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("./Debug/") + name + ".png", displayMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}


void displayFloat(std::string name, float *data, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);

	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
	}

	float min = 10000000000000000.f;
	float max = -10000000000000.0f;
	for (int i = 0; i < width * height; i++) {
		if (min > h_data[i]) min = h_data[i];
		if (max < h_data[i]) max = h_data[i];
	}
	printf("[%s]min: %.8f, max: %.8f\n", name, min, max);

	cv::Mat displayMat(height, width, CV_32FC1, h_data);
#ifdef SAVE_IMAGES
	cv::imwrite(std::string("./Debug/") + name + ".exr", displayMat);
#else
	cv::imshow(name, cv::abs(displayMat) * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}

void displayUpdateFloat3(std::string name, float3 *data, float3 *delta, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float3 *h_data = (float3 *)malloc(sizeof(float3) * width * height);
	float3 *h_delta = (float3 *)malloc(sizeof(float3) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_delta, delta, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float3) * width * height);
		cudaMemcpy(h_delta, delta, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}

	cv::Mat displayMat(height, width, CV_32FC3);
	cv::Mat deltaMat(height, width, CV_32FC3);

	memcpy(displayMat.data, h_data, sizeof(float3) * width * height);
	memcpy(deltaMat.data, h_delta, sizeof(float3) * width * height);

	displayMat = displayMat + deltaMat;

	cv::cvtColor(displayMat, displayMat, CV_BGR2RGB);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".exr", displayMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}

void displayUpdateFloat3SavePNG(std::string name, float3 *data, float3 *delta, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float3 *h_data = (float3 *)malloc(sizeof(float3) * width * height);
	float3 *h_delta = (float3 *)malloc(sizeof(float3) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_delta, delta, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float3) * width * height);
		cudaMemcpy(h_delta, delta, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}

	cv::Mat displayMat(height, width, CV_32FC3);
	cv::Mat deltaMat(height, width, CV_32FC3);
	cv::Mat printMat(height, width, CV_8UC3);

	memcpy(displayMat.data, h_data, sizeof(float3) * width * height);
	memcpy(deltaMat.data, h_delta, sizeof(float3) * width * height);

	displayMat = displayMat + deltaMat;

	cv::cvtColor(displayMat, displayMat, CV_BGR2RGB);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			cv::Vec3f color = displayMat.at<cv::Vec3f>(y, x);
			printMat.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0] * 255.f, color[1] * 255.f, color[2] * 255.f);
		}
	}

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", printMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}

void displayFloat3SavePNG(std::string name, float3 *data, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float3 *h_data = (float3 *)malloc(sizeof(float3) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float3) * width * height);
	}

	cv::Mat displayMat(height, width, CV_32FC3);
	cv::Mat printMat(height, width, CV_8UC3);
	memcpy(displayMat.data, h_data, sizeof(float3) * width * height);

	cv::cvtColor(displayMat, displayMat, CV_BGR2RGB);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			cv::Vec3f color = displayMat.at<cv::Vec3f>(y, x);
			printMat.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0] * 255.f, color[1] * 255.f, color[2] * 255.f);
		}
	}

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", printMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}

void displayFloat3(std::string name, float3 *data, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float3 *h_data = (float3 *)malloc(sizeof(float3) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float3) * width * height);
	}

	cv::Mat displayMat(height, width, CV_32FC3);
	memcpy(displayMat.data, h_data, sizeof(float3) * width * height);

	cv::cvtColor(displayMat, displayMat, CV_BGR2RGB);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".exr", displayMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	free(h_data);
}


void displayFloat4(std::string name, float4 *data, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float4 *h_data = (float4 *)malloc(sizeof(float4) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float4) * width * height);
	}

	cv::Mat displayMat(height, width, CV_32FC4);
	memcpy(displayMat.data, h_data, sizeof(float4) * width * height);

	cv::cvtColor(displayMat, displayMat, CV_BGRA2RGBA);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".exr", displayMat);
#else
	cv::imshow(name, displayMat * multiplier);
	cv::waitKey(0);
#endif
	displayMat.release();
	free(h_data);
}

void displayDifference(std::string name, float *data1, float *data2, bool *mask, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float *h_data1 = (float *)malloc(sizeof(float) * width * height);
	float *h_data2 = (float *)malloc(sizeof(float) * width * height);
	bool *h_mask = (bool *)malloc(sizeof(bool) * width * height);

	float sum = 0.0f;
	unsigned int numValidPixel = 0;

	if (isGPU) {
		cudaMemcpy(h_data1, data1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_data2, data2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mask, mask, sizeof(bool) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data1, data1, sizeof(float) * width * height);
		memcpy(h_data2, data2, sizeof(float) * width * height);
		memcpy(h_mask, mask, sizeof(bool) * width * height);
	}

	cv::Mat displayMat(height, width, CV_8UC3);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			if (h_mask[h * width + w]) {
				float diff = h_data1[h * width + w] - h_data2[h * width + w];
				sum += abs(diff);
				numValidPixel++;

				uchar3 value;
				value.x = 0; value.y = 0, value.z = 0;
				if (diff > 0) value.x = diff * multiplier;
				else value.y = diff * multiplier;
				displayMat.at<uchar3>(h, w) = value;
			}
		}
	}

	printf("Sum difference: %.5f, Average difference: %.5f\n", sum, sum / (float)numValidPixel);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", displayMat);
#else
	cv::imshow(name, displayMat);
	cv::waitKey(0);
#endif
	free(h_data1);
	free(h_data2);
	displayMat.release();
}

void displayUpdateDifference(std::string name, float *data1, float *data2, float *delta, bool *mask, int width, int height, bool isGPU, float multiplier = 1.0f) {
	float *h_data1 = (float *)malloc(sizeof(float) * width * height);
	float *h_data2 = (float *)malloc(sizeof(float) * width * height);
	float *h_delta = (float *)malloc(sizeof(float) * width * height);
	bool *h_mask = (bool *)malloc(sizeof(bool) * width * height);

	float sum = 0.0f;
	unsigned int numValidPixel = 0;

	if (isGPU) {
		cudaMemcpy(h_data1, data1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_data2, data2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_delta, delta, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mask, mask, sizeof(bool) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data1, data1, sizeof(float) * width * height);
		memcpy(h_data2, data2, sizeof(float) * width * height);
		memcpy(h_delta, delta, sizeof(float) * width * height);
		memcpy(h_mask, mask, sizeof(bool) * width * height);
	}

	cv::Mat displayMat(height, width, CV_8UC3);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			uchar3 zero;
			zero.x = 0; zero.y = 0; zero.z = 0;
			displayMat.at<uchar3>(h, w) = zero;
			if (h_mask[h * width + w]) {
				float diff = h_data1[h * width + w] - (h_data2[h * width + w] + h_delta[h * width + w]);
				sum += abs(diff);
				numValidPixel++;

				uchar3 value;
				value.x = 0; value.y = 0, value.z = 0;
				if (diff > 0) value.x = diff * multiplier;
				else value.y = diff * multiplier;
				displayMat.at<uchar3>(h, w) = value;
			}
		}
	}

	printf("Sum difference: %.5f, Average difference: %.5f\n", sum, sum / (float)numValidPixel);
#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", displayMat);
#else
	cv::imshow(name, displayMat);
	cv::waitKey(0);
#endif
	free(h_data1);
	free(h_data2);
	free(h_delta);
	displayMat.release();
}

void displayDepthNormal(std::string name, float *data, int width, int height, float fx, float fy, float cx, float cy, int nLevel, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
	}

	char *depthNormal = new char[width * height * 3];
	memset(depthNormal, 0, sizeof(char) * width * height * 3);
	int LevelParams = pow(2, nLevel) * 2;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float d00 = h_data[(y * width + x)];
			float dn0 = h_data[(y * width + x - 1)];
			float d0n = h_data[((y - 1) * width + x)];
			float dp0 = h_data[(y * width + x + 1)];
			float d0p = h_data[((y + 1) * width + x)];
			if (d00 > 0 && dn0 > 0 && d0n > 0 && dp0 > 0 && d0p > 0) {
				float3 p_n0 = make_float3((float(x - 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dn0;
				float3 p_0n = make_float3((float(x) - cx) / fx, (float(y - 1) - cy) / fy, 1.0f) * d0n;
				float3 p_p0 = make_float3((float(x + 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dp0;
				float3 p_0p = make_float3((float(x) - cx) / fx, (float(y + 1) - cy) / fy, 1.0f) * d0p;

				float3 n = cross((p_0n - p_0p), (p_n0 - p_p0));

				float l = length(n);
				if (l > 0) {
					n /= l;
					int idx = y * width + x;
					depthNormal[3 * idx + 0] = (n.x + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 1] = (-n.y + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 2] = (-n.z + 1.0f) / 2.f * 255;
				}
			}
		}
	}
	printf("Finish cal\n");
	getchar();

	cv::Mat depthNormalMat(height, width, CV_8UC3, depthNormal);
	cv::cvtColor(depthNormalMat, depthNormalMat, CV_RGB2BGR);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", depthNormalMat);
#else
	printf("prepare show\n");
	getchar();
	cv::imshow(name, depthNormalMat);
	//cv::waitKey(0);
	printf("showing\n");
	getchar();
#endif
	free(h_data);
	free(depthNormal);
}

void displayDepthNormal(std::string name, float *data, float *iterData, int width, int height, float fx, float fy, float cx, float cy, int nLevel, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	float *h_data2 = (float *)malloc(sizeof(float) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_data2, iterData, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
		memcpy(h_data2, iterData, sizeof(float) * width * height);
	}

	char *depthNormal = new char[width * height * 3];
	memset(depthNormal, 0, sizeof(char) * width * height * 3);
	int LevelParams = pow(2, nLevel) * 2;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float d00 = h_data[(y * width + x)] + h_data2[(y * width + x)];
			float dn0 = h_data[(y * width + x - 1)] + h_data2[(y * width + x)];
			float d0n = h_data[((y - 1) * width + x)] + h_data2[((y - 1) * width + x)];
			float dp0 = h_data[(y * width + x + 1)] + h_data2[(y * width + x + 1)];
			float d0p = h_data[((y + 1) * width + x)] + h_data2[((y + 1) * width + x)];
			if (d00 > 0 && dn0 > 0 && d0n > 0 && dp0 > 0 && d0p > 0) {
				float3 p_n0 = make_float3((float(x - 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dn0;
				float3 p_0n = make_float3((float(x) - cx) / fx, (float(y - 1) - cy) / fy, 1.0f) * d0n;
				float3 p_p0 = make_float3((float(x + 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dp0;
				float3 p_0p = make_float3((float(x) - cx) / fx, (float(y + 1) - cy) / fy, 1.0f) * d0p;

				float3 n = cross((p_0n - p_0p) / float(LevelParams), (p_n0 - p_p0) / float(LevelParams));

				float l = length(n);
				if (l > 0) {
					n /= l;
					int idx = y * width + x;
					depthNormal[3 * idx + 0] = (n.x + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 1] = (-n.y + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 2] = (-n.z + 1.0f) / 2.f * 255;
				}
			}
		}
	}

	cv::Mat depthNormalMat(height, width, CV_8UC3, depthNormal);
	cv::cvtColor(depthNormalMat, depthNormalMat, CV_RGB2BGR);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", depthNormalMat);
	printf("Finish to display iterative normal\n");
#else
	printf("Finish to display iterative normal\n");
	cv::imshow(name, depthNormalMat);
	cv::waitKey(0);
#endif
	free(h_data);
	free(h_data2);
	free(depthNormal);
}


float computeShading2(float *lightCoeffs, float3 n) {
	float sum = 0;

	sum += lightCoeffs[0];
	sum += lightCoeffs[1] * n.y;
	sum += lightCoeffs[2] * n.z;
	sum += lightCoeffs[3] * n.x;
	sum += lightCoeffs[4] * n.x * n.y;
	sum += lightCoeffs[5] * n.y * n.z;
	sum += lightCoeffs[6] * (-n.x * n.x - n.y * n.y + 2.f * n.z * n.z);
	sum += lightCoeffs[7] * n.z * n.x;
	sum += lightCoeffs[8] * (n.x * n.x - n.y * n.y);

	return sum;
}

float computeDeltaShading(float *lightCoeffs, float *lightDelta, float3 n) {
	float sum = 0;

	sum += (lightCoeffs[0] + lightDelta[0]);
	sum += (lightCoeffs[1] + lightDelta[1]) * n.y;
	sum += (lightCoeffs[2] + lightDelta[2]) * n.z;
	sum += (lightCoeffs[3] + lightDelta[3]) * n.x;
	sum += (lightCoeffs[4] + lightDelta[4]) * n.x * n.y;
	sum += (lightCoeffs[5] + lightDelta[5]) * n.y * n.z;
	sum += (lightCoeffs[6] + lightDelta[6]) * (-n.x * n.x - n.y * n.y + 2.f * n.z * n.z);
	sum += (lightCoeffs[7] + lightDelta[7]) * n.z * n.x;
	sum += (lightCoeffs[8] + lightDelta[8]) * (n.x * n.x - n.y * n.y);

	return sum;
}

//void displayDepthRender(std::string name, float *data, float *albedo, float *light, int width, int height, float fx, float fy, float cx, float cy, int nLevel, bool isGPU) {
//	float *h_data = (float *)malloc(sizeof(float) * width * height);
//	float *h_albedo = (float *)malloc(sizeof(float) * width * height * 3);
//	float *h_light = (float *)malloc(sizeof(float) * 9);
//	if (isGPU) {
//		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
//		cudaMemcpy(h_albedo, albedo, sizeof(float) * width * height * 3, cudaMemcpyDeviceToHost);
//		cudaMemcpy(h_light, light, sizeof(float) * 9, cudaMemcpyDeviceToHost);
//	}
//	else {
//		memcpy(h_data, data, sizeof(float) * width * height);
//		memcpy(h_albedo, albedo, sizeof(float) * width * height * 3);
//		memcpy(h_light, light, sizeof(float) * 9);
//	}
//
//	int LevelParams = pow(2, nLevel);
//	LevelParams = LevelParams * LevelParams;
//	char *shading = new char[width * height * 3];
//	memset(shading, 0, sizeof(char) * width * height * 3);
//
//	float *h_data2 = (float *)malloc(sizeof(float) * width * height);
//	for (int y = 0; y < height - 1; y++) {
//		for (int x = 0; x < width - 1; x++) {
//			float d00 = h_data[(y * width + x)];
//			float dp0 = h_data[(y * width + x + 1)];
//			float d0p = h_data[((y + 1) * width + x)];
//			float dpp = h_data[((y + 1) * width + x + 1)];
//			if (d00 > 0 && dp0 > 0 && d0p > 0 && dpp > 0) {
//				h_data2[y * width + x] = (d00 + dp0 + d0p + dpp) / 4.0f;
//			}
//		}
//	}
//
//	for (int y = 1; y < height; y++) {
//		for (int x = 1; x < width; x++) {
//			float d00 = h_data2[(y * width + x)];
//			float dn0 = h_data2[(y * width + x - 1)];
//			float d0n = h_data2[((y - 1) * width + x)];
//			if (d00 > 0 && dn0 > 0 && d0n > 0) {
//				float3 n;
//				n.x = d0n * (d00 - dn0) / fy / float(LevelParams);
//				n.y = dn0 * (d00 - d0n) / fx / float(LevelParams);
//				n.z = ((n.x * (cx - x) / fx) + (n.y * (cy - y) / fy) - (dn0 * d0n / fx / fy)) / float(LevelParams);
//
//				float l = length(n);
//				if (l > 0) {
//					n /= l;
//					float shade = computeShading(h_light, n);
//
//					int idx = y * width + x;
//
//					shading[3 * idx + 0] = h_albedo[3 * idx + 0] * shade * 255;
//					shading[3 * idx + 1] = h_albedo[3 * idx + 1] * shade * 255;
//					shading[3 * idx + 2] = h_albedo[3 * idx + 2] * shade * 255;
//				}
//			}
//		}
//	}
//
//	free(h_data2);
//	cv::Mat depthShadingMat(height, width, CV_8UC3, shading);
//	cv::cvtColor(depthShadingMat, depthShadingMat, CV_RGB2BGR);
//	
//
//#ifdef SAVE_IMAGES
//	cv::imwrite(std::string("Debug/") + name + ".png", depthShadingMat);
//#else
//	cv::imshow(name, depthShadingMat);
//	cv::waitKey(0);
//#endif
//	free(h_data);
//	free(h_albedo);
//	free(h_light);
//	free(shading);
//}

void displayDepthRender(std::string name, float *data, float3 *albedo, float *light, int width, int height, float fx, float fy, float cx, float cy, int nLevel, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	float3 *h_albedo = (float3 *)malloc(sizeof(float3) * width * height);
	float *h_light = (float *)malloc(sizeof(float) * 9);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_albedo, albedo, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_light, light, sizeof(float) * 9, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
		memcpy(h_albedo, albedo, sizeof(float3) * width * height);
		memcpy(h_light, light, sizeof(float) * 9);
	}

	int LevelParams = pow(2, nLevel) * 2;
	char *shading = new char[width * height * 3];
	memset(shading, 0, sizeof(char) * width * height * 3);
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float d00 = h_data[(y * width + x)];
			float dn0 = h_data[(y * width + x - 1)];
			float d0n = h_data[((y - 1) * width + x)];
			float dp0 = h_data[(y * width + x + 1)];
			float d0p = h_data[((y + 1) * width + x)];
			if (d00 > 0 && dn0 > 0 && d0n > 0 && dp0 > 0 && d0p > 0) {

				float3 p_n0 = make_float3((float(x - 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dn0;
				float3 p_0n = make_float3((float(x) - cx) / fx, (float(y - 1) - cy) / fy, 1.0f) * d0n;
				float3 p_p0 = make_float3((float(x + 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dp0;
				float3 p_0p = make_float3((float(x) - cx) / fx, (float(y + 1) - cy) / fy, 1.0f) * d0p;

				float3 n = cross((p_0n - p_0p) / float(LevelParams), (p_n0 - p_p0) / float(LevelParams));

				float l = length(n);
				if (l > 0) {
					n /= l;
					float shade = computeShading2(h_light, n);

					int idx = y * width + x;

					shading[3 * idx + 0] = min(max(0.0f, h_albedo[idx].x * shade), 1.0f) * 255;
					shading[3 * idx + 1] = min(max(0.0f, h_albedo[idx].y * shade), 1.0f) * 255;
					shading[3 * idx + 2] = min(max(0.0f, h_albedo[idx].z * shade), 1.0f) * 255;
				}
			}
		}
	}
	cv::Mat depthShadingMat(height, width, CV_8UC3, shading);
	cv::cvtColor(depthShadingMat, depthShadingMat, CV_RGB2BGR);


#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", depthShadingMat);
#else
	cv::imshow(name, depthShadingMat);
	cv::waitKey(0);
#endif
	free(h_data);
	free(h_albedo);
	free(h_light);
	free(shading);
}

void displayUpdateDepthRender(std::string name, float *data, float *delta, float3 *albedo, float3 *albedo_delta, float *light, float *light_delta, int width, int height, float fx, float fy, float cx, float cy, int nLevel, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	float *h_delta = (float *)malloc(sizeof(float) * width * height);
	float3 *h_albedo = (float3 *)malloc(sizeof(float3) * width * height);
	float3 *h_albedo_delta = (float3 *)malloc(sizeof(float3) * width * height);
	float *h_light = (float *)malloc(sizeof(float) * 9);
	float *h_light_delta = (float *)malloc(sizeof(float) * 9);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_delta, delta, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_albedo, albedo, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_albedo_delta, albedo_delta, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_light, light, sizeof(float) * 9, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_light_delta, light_delta, sizeof(float) * 9, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
		memcpy(h_delta, delta, sizeof(float) * width * height);
		memcpy(h_albedo, albedo, sizeof(float3) * width * height);
		memcpy(h_albedo_delta, albedo_delta, sizeof(float3) * width * height);
		memcpy(h_light, light, sizeof(float) * 9);
		memcpy(h_light_delta, light_delta, sizeof(float) * 9);
	}

	char *shading = new char[width * height * 3];
	memset(shading, 0, sizeof(char) * width * height * 3);
	int LevelParams = pow(2, nLevel) * 2;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float d00 = h_data[(y * width + x)] + h_delta[(y * width + x)];
			float dn0 = h_data[(y * width + x - 1)] + h_delta[(y * width + x - 1)];
			float d0n = h_data[((y - 1) * width + x)] + h_delta[(y * width + x - 1)];
			float dp0 = h_data[(y * width + x + 1)] + h_delta[(y * width + x + 1)];
			float d0p = h_data[((y + 1) * width + x)] + h_delta[((y + 1) * width + x)];
			if (d00 > 0 && dn0 > 0 && d0n > 0 && dp0 > 0 && d0p > 0) {
				float3 p_n0 = make_float3((float(x - 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dn0;
				float3 p_0n = make_float3((float(x) - cx) / fx, (float(y - 1) - cy) / fy, 1.0f) * d0n;
				float3 p_p0 = make_float3((float(x + 1) - cx) / fx, (float(y) - cy) / fy, 1.0f) * dp0;
				float3 p_0p = make_float3((float(x) - cx) / fx, (float(y + 1) - cy) / fy, 1.0f) * d0p;

				float3 n = cross((p_0n - p_0p) / float(LevelParams), (p_n0 - p_p0) / float(LevelParams));

				float l = length(n);
				if (l > 0) {
					n /= l;
					float shade = computeDeltaShading(h_light, h_light_delta, n);
					//if (shade < 0) shade = 0.0f;
					//if (shade > 1) shade = 1.0f;
					int idx = y * width + x;

					float3 shadingRGB = clamp((h_albedo[idx] + h_albedo_delta[idx]) * shade, 0.0f, 1.0f);
					shading[3 * idx + 0] = shadingRGB.x * 255;
					shading[3 * idx + 1] = shadingRGB.y * 255;
					shading[3 * idx + 2] = shadingRGB.z * 255;
				}
			}
		}
	}
	cv::Mat depthShadingMat(height, width, CV_8UC3, shading);
	cv::cvtColor(depthShadingMat, depthShadingMat, CV_RGB2BGR);

#ifdef SAVE_IMAGES
	cv::imwrite(std::string("Debug/") + name + ".png", depthShadingMat);
#else
	cv::imshow(name, depthShadingMat);
	cv::waitKey(0);
#endif
	free(h_data);
	free(h_delta);
	free(h_albedo);
	free(h_albedo_delta);
	free(h_light);
	free(h_light_delta);
	free(shading);
}
