#ifndef __CUDADEBUG_H__
#define __CUDADEBUG_H__

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

template<typename T>
static void CheckRuntimeError(T error, const std::string& functionName, const std::string& fileName, const int line, const char* message)
{
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA ERROR:" << std::endl;
		std::cerr << "   file: " << fileName << std::endl;
		std::cerr << "   line: " << line << std::endl;
		std::cerr << "   func: " << functionName << std::endl;
		std::cerr << "   msg: " << message << std::endl;
		std::cerr << "   desc: " << cudaGetErrorString(error) << std::endl;
		throw std::runtime_error(message);

	}
}
#define CHECK_CUDA_ERROR(error, functionName, message) CheckRuntimeError( (error), functionName, __FILE__, __LINE__, (message) );

extern "C"
static void printFloat4x4(float4x4 a, std::string name){
	printf("%s\n", name.c_str());
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			printf("%f ", a(i,j));
		}
		printf("\n");
	}
}

static bool writeUint3Array(uint3 *d_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		uint3 *h_array = (uint3*)malloc(sizeof(uint3) * n);

		cudaMemcpy(h_array, d_array, sizeof(uint3) * n, cudaMemcpyDeviceToHost);

		int i;

		printf("write uint3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write uint3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u %u %u \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		free(h_array);

		return true;

	}
	else return false;
}

static bool writeUint3ArrayHost(uint3 *h_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (h_array != NULL) {

		FILE *out = fopen(filename.data(), "w");

		int i;

		printf("write uint3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write uint3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u %u %u \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		return true;

	}
	else return false;
}

static bool writeFloat3Array(float3 *d_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		float3 *h_array = (float3*)malloc(sizeof(float3) * n);

		cudaMemcpy(h_array, d_array, sizeof(float3) * n, cudaMemcpyDeviceToHost);

		int i;

		printf("write float3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write float3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f %f %f \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		free(h_array);


		return true;

	}
	else return false;
}
static bool writeFloatArray(float *d_array, std::string filename, int n) {

	printf("write float array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		float *h_array = (float*)malloc(sizeof(float) * n);

		cudaMemcpy(h_array, d_array, sizeof(float) * n, cudaMemcpyDeviceToHost);

		int i;
    
		//print the number of vectors
		fprintf(out, "%d\n", n);

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f \n ", h_array[i]);

		printf("finished\n");

		free(h_array);

		fclose(out);

		return true;

	}
	else return false;
}



static bool writeFloatNArray(float *d_array, std::string filename, int n, int ch) {

  if (d_array != NULL) {

    FILE *out = fopen(filename.data(), "w");
    float *h_array = (float*)malloc(sizeof(float) * n * ch);

    cudaMemcpy(h_array, d_array, sizeof(float) * n *ch, cudaMemcpyDeviceToHost);

    int i;

    printf("write float array\n");

    //print the number of vectors
    fprintf(out, "%d\n", n);

    //print triangle indices
    for (i = 0; i < n; i++) {
      for (int j = 0; j < ch; j++)
        if(h_array[i*ch + j]>0.0001)
          fprintf(out, "%.4f  ", h_array[i*ch + j]);
      fprintf(out, "\n");
    }
    printf("finished\n");

    free(h_array);

    return true;

  }
  else return false;
}
static bool writeFloat3ArrayHost(float3 *h_array, std::string filename, int n) {

	if (h_array != NULL) {

		FILE *out = fopen(filename.data(), "w");

		int i;

		printf("write float3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write float3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f %f %f \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		return true;

	}
	else return false;
}

static bool writeUintArray(unsigned int *d_array, std::string filename, int n) {

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		unsigned int *h_array = (unsigned int*)malloc(sizeof(unsigned int) * n);

		cudaMemcpy(h_array, d_array, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		fprintf(out, "%d\n", n);

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u ", h_array[i]);


		free(h_array);

		return true;

	}
	else return false;
}

template <typename T>
static bool writeArray(const T *d_array, std::string filename, int n, int nch=1) {

	if (d_array != NULL) {

		std::ofstream out(filename, std::ofstream::out);
		T *h_array = (T*)malloc(sizeof(T) * n * nch);

		cudaMemcpy(h_array, d_array, sizeof(T) * n * nch, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		out << n <<std::endl;

		//print triangle indices
		for (i = 0; i < n; i++) {
			for (int j = 0; j < nch; j++)
				out << (T)h_array[i*nch + j]<<" ";
			out << std::endl;
		}
		free(h_array);

		return true;

	}
	else return false;
}
template <typename T>
static bool writeArray(const T *d_array, std::string filename, int w, int h, int nch ) {

	if (d_array != NULL) {
		w = w/2;
		int n = w*h; 

		std::ofstream out(filename, std::ofstream::out);
		T *h_array = (T*)malloc(sizeof(T) * n * nch);

		cudaMemcpy(h_array, d_array, sizeof(T) * n * nch, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		out << n << std::endl;

		//print triangle indices
		for (i = 0; i < h; i++) {
			for (int k = 0; k < w; k++) {
				for (int j = 0; j < nch; j++)
					out << (T)h_array[(i*w + k)*nch + j] << " ";
			}
			out << std::endl;
		}
		free(h_array);

		return true;

	}
	else return false;
}

template <typename T>
static bool writeArrayHost(const T *h_array, std::string filename, int n, int nch = 1) {

	if (h_array != NULL) {

		std::ofstream out(filename, std::ofstream::out);
	
		int i;

		//print the number of vectors
		out << n << std::endl;

		//print triangle indices
		for (i = 0; i < n; i++) {
			for (int j = 0; j < nch; j++)
				out << h_array[i*nch + j] << " ";
			out << std::endl;
		}

		return true;

	}
	else return false;
}

static bool displayFloatMat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);
	a = a * scale + offset;

	cv::imshow(windowname, a);
	//cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatMatSingle(float *d_mat, std::string windowname, int width, int height, int nch, int selectedch, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp = (float*)malloc(sizeof(float) * width * height*nch);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height*nch, cudaMemcpyDeviceToHost);

	for (int i = 0; i < width*height; i++) 
		tmp[i] = tmp[i*nch + selectedch];

	cv::Mat a(height, width, CV_32FC1, tmp);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayDiffFloatMat(float *d_mat1, float *d_mat2, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp1 = (float*)malloc(sizeof(float) * width * height);
	float *tmp2 = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp1, d_mat1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp2, d_mat2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	
	cv::Mat a1(height, width, CV_32FC1, tmp1);
	cv::Mat a2(height, width, CV_32FC1, tmp2);

	cv::imshow(windowname, (a1-a2) * scale + offset);
//	cv::waitKey(0);

	free(tmp1);
	free(tmp2);

	return true;

}

static bool writeDeltaFloat(float *delta_rot, float *delta_trans, std::string filename, int frame_cnt) {

	if (frame_cnt == 2) {
		std::ofstream out(filename, std::ofstream::out);

		out << delta_rot[0] << " " << delta_rot[1] << " "  << delta_rot[2] << " " << delta_trans[0] << " " << delta_trans[1] << " " << delta_trans[2] << std::endl;
	}
	else {
		std::ofstream out(filename, std::ofstream::out | std::ofstream::app);

		out << delta_rot[0] << " " << delta_rot[1] << " " << delta_rot[2] << " " << delta_trans[0] << " " << delta_trans[1] << " " << delta_trans[2] << std::endl;
	}
	return true;
}

static bool writeDiffFloatMat(float *d_mat1, float *d_mat2, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp1 = (float*)malloc(sizeof(float) * width * height);
	float *tmp2 = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp1, d_mat1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp2, d_mat2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a1(height, width, CV_32FC1, tmp1);
	cv::Mat a2(height, width, CV_32FC1, tmp2);

	cv::Mat b = (a1 - a2) * scale + offset;
	b *= 255;

	cv::imwrite(windowname, b);
	//	cv::waitKey(0);

	free(tmp1);
	free(tmp2);

	return true;

}
static bool displayNormalizedFloatMat(float *d_mat, std::string windowname, int width, int height) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);

	double v_min, v_max;
	cv::minMaxLoc(a, &v_min, &v_max); 

	cv::imshow(windowname, (a-v_min)/(v_max-v_min));
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloat4Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 4);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC4, tmp);

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{

			((float*)a.data)[4 * (i*a.cols + j) + 3] = 1;

		}
	}

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatMatHost(float *h_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	cv::Mat a(height, width, CV_32FC1, h_mat);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	return true;

}


static bool displayFloat2Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 2);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 2, cudaMemcpyDeviceToHost);

  printf("%d %d\n", width, height);

	cv::Mat a(height, width, CV_32FC2, tmp);
	cv::Mat b(height, width, CV_32FC1);

  a = scale * a + offset;

	memset(b.data, 0, sizeof(float) * width *height);

	std::vector<cv::Mat> mats;
	
	cv::split(a, mats);

	mats.push_back(b);

	cv::Mat tmpmat;

	cv::merge(mats, tmpmat);

	cv::imshow(windowname, tmpmat);
//	cv::waitKey(0);

	free(tmp);

	return true;

}


static bool displayInt2Mat(int *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

  int *tmp = (int*)malloc(sizeof(int) * width * height * 2);
  cudaMemcpy(tmp, d_mat, sizeof(int) * width * height * 2, cudaMemcpyDeviceToHost);

  cv::Mat a(height, width, CV_32SC2, tmp);
  cv::Mat b(height, width, CV_32SC1);


  a.convertTo(a, CV_32FC2);
  b.convertTo(b, CV_32FC1);

  memset(b.data, 0, sizeof(int) * width * height);

  std::vector<cv::Mat> mats;

  cv::split(a, mats);

  mats.push_back(b);

  cv::Mat tmpmat;

  cv::merge(mats, tmpmat);


  cv::imshow(windowname, scale *  tmpmat + offset);
  cv::waitKey(0);

  free(tmp);

  return true;

}


static bool displayIntMat(int *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	int *tmp = (int*)malloc(sizeof(int) * width * height );
	cudaMemcpy(tmp, d_mat, sizeof(int) * width * height , cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32SC1, tmp);

	a.convertTo(a, CV_32FC1);
	
	cv::imshow(windowname, scale *  a + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloat3MatHost(float *h_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {
	

	cv::Mat a(height, width, CV_32FC3, h_mat);
	cv::imshow(windowname, scale *  a + offset);
	cv::waitKey(0);

	return true;

}


static bool displayFloat3Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height*3);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height*3, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC3, tmp);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatNMat(float *d_mat, std::string windowname, int width, int height, int nch, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * nch);
	float *tmp_single = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * nch, cudaMemcpyDeviceToHost);

	int i, j;

	for (int k = 0; k < nch; k++) {

		for (j = 0; j < height; j++) {
			for (i = 0; i < width; i++) {

				int index = j * width + i;

				tmp_single[index] = tmp[index*nch + k];
			}

		}
		cv::Mat b(height, width, CV_32FC1, tmp_single);
		cv::imshow(windowname, scale * b + offset);
		cv::waitKey(0);

	}

	free(tmp);
	free(tmp_single);

	return true;

}

static bool writeFloatMat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);


	std::cout << windowname << std::endl;
	a = a * scale + offset;
	a *= 255;
	cv::imwrite(windowname, a);


	free(tmp);

	return true;

}

static bool writeNormalizedFloatMat(float *d_mat, std::string windowname, int width, int height) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);

	double v_min, v_max;
	cv::minMaxLoc(a, &v_min, &v_max);

	cv::imwrite(windowname, 255 *(a - v_min) / (v_max - v_min));

	free(tmp);

	return true;

}

static bool writeFloat4Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 4);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC4, tmp);

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{

			((float*)a.data)[4*(i*a.cols + j) + 3] = 1;

		}
	}
	a =  (a * scale + offset);

	cv::imwrite(windowname, a);

	free(tmp);

	return true;

}

static bool writeBooleanMat(bool *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	bool *tmp = (bool*)malloc(sizeof(bool) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(bool) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_8UC3);

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			a.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			if (tmp[i * width + j]) {
				a.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	cv::imwrite(windowname, a);

	free(tmp);

	return true;

}

static void writeAlbedo(float4 *data, std::string name, int width, int height, bool isGPU, float scale = 1.0f) {
	float4 *h_data = (float4 *)malloc(sizeof(float4) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float4) * width * height);
	}

	char *albedo = new char[width * height * 3];
	memset(albedo, 0, sizeof(char) * width * height * 3);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = y * width + x;
			albedo[3 * idx + 0] = h_data[idx].z * scale * 255;
			albedo[3 * idx + 1] = h_data[idx].y * scale * 255;
			albedo[3 * idx + 2] = h_data[idx].x * scale * 255;
		}
	}

	cv::Mat albedoMat(height, width, CV_8UC3, albedo);

	cv::imwrite(std::string("Debug/") + name + ".png", albedoMat);

	free(h_data);
	free(albedo);
}

static void writeNormalAndColor(float4 *data, std::string name, int width, int height, bool isGPU, float scale = 1.0f) {
	float4 *h_data = (float4 *)malloc(sizeof(float4) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float4) * width * height);
	}

	unsigned char *albedo = new unsigned char[width * height * 3];
	memset(albedo, 0, sizeof(unsigned char) * width * height * 3);
	unsigned char *color = new unsigned char[width * height];
	memset(color, 0, sizeof(unsigned char) * width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = y * width + x;
			albedo[3 * idx + 0] = h_data[idx].z * scale * 255;
			albedo[3 * idx + 1] = h_data[idx].y * scale * 255;
			albedo[3 * idx + 2] = h_data[idx].x * scale * 255;
			color[idx] = h_data[idx].w * 255.f;
		}
	}

	cv::Mat albedoMat(height, width, CV_8UC3, albedo);
	cv::Mat colorMat(height, width, CV_8UC1, color);
	cv::imwrite(std::string("Debug/") + name + ".png", albedoMat);
	cv::imwrite(std::string("Debug/") + name + "_color.png", colorMat);

	free(h_data);
	free(color);
	free(albedo);
}

static void writeNormal(float4 *data, std::string name, int width, int height, bool isGPU, float scale = 1.0f) {
	float4 *h_data = (float4 *)malloc(sizeof(float4) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float4) * width * height);
	}

	char *albedo = new char[width * height * 3];
	memset(albedo, 0, sizeof(char) * width * height * 3);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = y * width + x;
			albedo[3 * idx + 0] = (-h_data[idx].z + 1.0f) * 0.5f * 255.f;
			albedo[3 * idx + 1] = (-h_data[idx].y + 1.0f) * 0.5f * 255.f;
			albedo[3 * idx + 2] = (h_data[idx].x + 1.0f) * 0.5f * 255.f;
		}
	}

	cv::Mat albedoMat(height, width, CV_8UC3, albedo);

	cv::imwrite(std::string("Debug/") + name + ".png", albedoMat);

	free(h_data);
	free(albedo);
}

static void writeDepthNormal(float *data, std::string name, int width, int height, float fx, float fy, float cx, float cy, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
	}

	char *depthNormal = new char[width * height * 3];
	memset(depthNormal, 0, sizeof(char) * width * height * 3);

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
					depthNormal[3 * idx + 0] = (-n.z + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 1] = (-n.y + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 2] = (n.x + 1.0f) / 2.f * 255;
				}
			}
		}
	}

	cv::Mat depthNormalMat(height, width, CV_8UC3, depthNormal);

	cv::imwrite(std::string("Debug/") + name + ".png", depthNormalMat);

	free(h_data);
	free(depthNormal);
}

static void writeDepth4Normal(float4 *data, std::string name, int width, int height, float fx, float fy, float cx, float cy, bool isGPU) {
	float4 *h_data = (float4 *)malloc(sizeof(float4) * width * height);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float4) * width * height);
	}

	char *depthNormal = new char[width * height * 3];
	memset(depthNormal, 0, sizeof(char) * width * height * 3);

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float3 p_n0 = make_float3(h_data[int(y + 0) * width + int(x - 1)]);
			float3 p_0n = make_float3(h_data[int(y - 1) * width + int(x + 0)]);
			float3 p_p0 = make_float3(h_data[int(y + 0) * width + int(x + 1)]);
			float3 p_0p = make_float3(h_data[int(y + 1) * width + int(x + 0)]);
			if (p_n0.z > 0.f && p_0n.z > 0.f, p_p0.z > 0.f, p_0p.z > 0.f) {
				float3 n = cross((p_0n - p_0p), (p_n0 - p_p0));

				float l = length(n);
				if (l > 0) {
					n /= l;
					int idx = y * width + x;
					depthNormal[3 * idx + 0] = (-n.z + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 1] = (-n.y + 1.0f) / 2.f * 255;
					depthNormal[3 * idx + 2] = (n.x + 1.0f) / 2.f * 255;
				}
			}
		}
	}

	cv::Mat depthNormalMat(height, width, CV_8UC3, depthNormal);

	cv::imwrite(std::string("Debug/") + name + ".png", depthNormalMat);

	free(h_data);
	free(depthNormal);
}

static float computeShading(float *lightCoeffs, float3 n) {
	float sum = 0.f;

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

static void writeDepthRender(std::string name, float *data, float4 *albedo, float *light, int width, int height, float fx, float fy, float cx, float cy, bool isGPU) {
	float *h_data = (float *)malloc(sizeof(float) * width * height);
	float4 *h_albedo = (float4 *)malloc(sizeof(float4) * width * height);
	float *h_light = (float *)malloc(sizeof(float) * 9);
	if (isGPU) {
		cudaMemcpy(h_data, data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_albedo, albedo, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_light, light, sizeof(float) * 9, cudaMemcpyDeviceToHost);
	}
	else {
		memcpy(h_data, data, sizeof(float) * width * height);
		memcpy(h_albedo, albedo, sizeof(float4) * width * height);
		memcpy(h_light, light, sizeof(float) * 9);
	}

	unsigned char *shading = new unsigned char[width * height * 3];
	memset(shading, 0, sizeof(unsigned char) * width * height * 3);
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
					float shade = 0.5f;// computeShading(h_light, n);

					int idx = y * width + x;

					shading[3 * idx + 0] = min(max(0.0f, h_albedo[idx].z * shade), 1.0f) * 255.f;
					shading[3 * idx + 1] = min(max(0.0f, h_albedo[idx].y * shade), 1.0f) * 255.f;
					shading[3 * idx + 2] = min(max(0.0f, h_albedo[idx].x * shade), 1.0f) * 255.f;
				}
			}
		}
	}
	cv::Mat depthShadingMat(height, width, CV_8UC3, shading);

	//cv::imshow(name, depthShadingMat);
	//cv::waitKey(0);
	cv::imwrite(std::string("Debug/") + name + ".png", depthShadingMat);
	free(h_data);
	free(h_albedo);
	free(h_light);
	free(shading);
}
#endif