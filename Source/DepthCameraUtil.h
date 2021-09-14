#pragma once


#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

#include "CUDADepthCameraParams.h"


extern "C" void updateConstantDepthCameraParams(const DepthCameraParams& params);
extern __constant__ DepthCameraParams c_depthCameraParams;


struct DepthCameraData {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
	DepthCameraData() {
		d_depthData = NULL;
		d_colorData = NULL;
		d_lightData = NULL;
		d_depthMask = NULL;
		d_rhoDData = NULL;
		d_normalData = NULL;
		d_inputNormalData = NULL;
		d_depthArray = NULL;
		d_colorArray = NULL;
		d_rhoDArray = NULL;
		d_normalArray = NULL;
	}

	__host__
	void alloc(const DepthCameraParams& params) { //! todo resizing???
		cutilSafeCall(cudaMalloc(&d_depthData, sizeof(float) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_depthMask, sizeof(float) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_colorData, sizeof(float4) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_rhoDData, sizeof(float4) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_normalData, sizeof(float4) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_inputNormalData, sizeof(float4) * params.m_imageWidth * params.m_imageHeight));
		cutilSafeCall(cudaMalloc(&d_lightData, sizeof(float) * 9));
		h_depthChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&d_depthArray, &h_depthChannelDesc, params.m_imageWidth, params.m_imageHeight));
		h_colorChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&d_colorArray, &h_colorChannelDesc, params.m_imageWidth, params.m_imageHeight));
		h_rhoDChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&d_rhoDArray, &h_rhoDChannelDesc, params.m_imageWidth, params.m_imageHeight));
		h_normalChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&d_normalArray, &h_normalChannelDesc, params.m_imageWidth, params.m_imageHeight));
	}

	__host__
	void updateParams(const DepthCameraParams& params) {
		updateConstantDepthCameraParams(params);
	}

	__host__
	void free() {
		if (d_depthData) cutilSafeCall(cudaFree(d_depthData));
		if (d_depthMask) cutilSafeCall(cudaFree(d_depthMask));
		if (d_colorData) cutilSafeCall(cudaFree(d_colorData));
		if (d_rhoDData) cutilSafeCall(cudaFree(d_rhoDData));
		if (d_normalData) cutilSafeCall(cudaFree(d_normalData));
		if (d_inputNormalData) cutilSafeCall(cudaFree(d_inputNormalData));
		if (d_lightData) cutilSafeCall(cudaFree(d_lightData));
		if (d_depthArray) cutilSafeCall(cudaFreeArray(d_depthArray));
		if (d_colorArray) cutilSafeCall(cudaFreeArray(d_colorArray));
		if (d_rhoDArray) cutilSafeCall(cudaFreeArray(d_rhoDArray));
		if (d_normalArray) cutilSafeCall(cudaFreeArray(d_normalArray));

		d_depthData = NULL;
		d_depthMask = NULL;
		d_colorData = NULL;
		d_rhoDData = NULL;
		d_normalData = NULL;
		d_inputNormalData = NULL;
		d_lightData = NULL;
		d_depthArray = NULL;
		d_colorArray = NULL;
		d_rhoDArray = NULL;
		d_normalArray = NULL;
	}


	/////////////////
	// Device part //
	/////////////////

	static inline const DepthCameraParams& params() {
		return c_depthCameraParams;
	}

		///////////////////////////////////////////////////////////////
		// Camera to Screen
		///////////////////////////////////////////////////////////////

	__device__
	static inline float2 cameraToKinectScreenFloat(const float3& pos)	{
		//return make_float2(pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx, c_depthCameraParams.my - pos.y*c_depthCameraParams.fy/pos.z);
		return make_float2(
			pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx,			
			pos.y*c_depthCameraParams.fy/pos.z + c_depthCameraParams.my);
	}

	__device__
	static inline int2 cameraToKinectScreenInt(const float3& pos)	{
		float2 pImage = cameraToKinectScreenFloat(pos);
		return make_int2(pImage + make_float2(0.5f, 0.5f));
	}

	__device__
	static inline uint2 cameraToKinectScreen(const float3& pos)	{
		int2 p = cameraToKinectScreenInt(pos);
		return make_uint2(p.x, p.y);
	}

	__device__
		static inline float cameraToKinectProjZ(float z) {
		return (z - c_depthCameraParams.m_sensorDepthWorldMin) / (c_depthCameraParams.m_sensorDepthWorldMax - c_depthCameraParams.m_sensorDepthWorldMin);
	}

	__device__
		static inline float depthInverse(float z) {
		return min(1.f, c_depthCameraParams.m_sensorDepthWorldMin / z);
	}

	__device__
	static inline float3 cameraToKinectProj(const float3& pos) {
		float2 proj = cameraToKinectScreenFloat(pos);

		float3 pImage = make_float3(proj.x, proj.y, pos.z);

		pImage.x = (2.0f*pImage.x - (c_depthCameraParams.m_imageWidth- 1.0f))/(c_depthCameraParams.m_imageWidth- 1.0f);
		//pImage.y = (2.0f*pImage.y - (c_depthCameraParams.m_imageHeight-1.0f))/(c_depthCameraParams.m_imageHeight-1.0f);
		pImage.y = ((c_depthCameraParams.m_imageHeight-1.0f) - 2.0f*pImage.y)/(c_depthCameraParams.m_imageHeight-1.0f);
		pImage.z = cameraToKinectProjZ(pImage.z);

		return pImage;
	}

		///////////////////////////////////////////////////////////////
		// Screen to Camera (depth in meters)
		///////////////////////////////////////////////////////////////

	__device__
		static inline float3 kinectDepthToSkeleton(uint ux, uint uy, float depth) {
		const float x = ((float)ux - c_depthCameraParams.mx) / c_depthCameraParams.fx;
		const float y = ((float)uy - c_depthCameraParams.my) / c_depthCameraParams.fy;
		//const float y = (c_depthCameraParams.my-(float)uy) / c_depthCameraParams.fy;
		return make_float3(depth*x, depth*y, depth);
	}

	__device__
		static inline float3 kinectDepthToSkeletonFloat(float xf, float yf, float depth) {
		const float x = (xf - c_depthCameraParams.mx) / c_depthCameraParams.fx;
		const float y = (yf - c_depthCameraParams.my) / c_depthCameraParams.fy;
		//const float y = (c_depthCameraParams.my-(float)uy) / c_depthCameraParams.fy;
		return make_float3(depth * x, depth * y, depth);
	}

		///////////////////////////////////////////////////////////////
		// RenderScreen to Camera -- ATTENTION ASSUMES [1,0]-Z range!!!!
		///////////////////////////////////////////////////////////////

	__device__
	static inline float kinectProjToCameraZ(float z) {
		return z * (c_depthCameraParams.m_sensorDepthWorldMax - c_depthCameraParams.m_sensorDepthWorldMin) + c_depthCameraParams.m_sensorDepthWorldMin;
	}

	// z has to be in [0, 1]
	__device__
		static inline float3 kinectProjToCamera(uint ux, uint uy, float z) {
		float fSkeletonZ = kinectProjToCameraZ(z);
		return kinectDepthToSkeleton(ux, uy, fSkeletonZ);
	}


	__device__
		static inline float3 kinectProjToCameraFloat(float x, float y, float z) {
		float fSkeletonZ = kinectProjToCameraZ(z);
		return kinectDepthToSkeletonFloat(x, y, fSkeletonZ);
	}

	__device__
	static inline bool isInCameraFrustumApprox(const float4x4& viewMatrixInverse, const float3& pos) {
		float3 pCamera = viewMatrixInverse * pos;
		float3 pProj = cameraToKinectProj(pCamera);
		//pProj *= 1.5f;	//TODO THIS IS A HACK FIX IT :)
		pProj *= 0.95;
		return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);  
	}

	float*		d_depthData;	//depth data of the current frame (in screen space):: TODO data allocation lives in RGBD Sensor
	float4*		d_colorData;
	float4*		d_rhoDData;
	float4*		d_normalData;
	float4*		d_inputNormalData;
	float*		d_lightData;
	float*		d_depthMask;
	//uchar4*		d_colorData;	//color data of the current frame (in screen space):: TODO data allocation lives in RGBD Sensor

	// cuda arrays for texture access
	cudaArray*	d_depthArray;
	cudaArray*	d_colorArray;
	cudaArray*	d_rhoDArray;
	cudaArray*	d_normalArray;
	cudaChannelFormatDesc h_depthChannelDesc;
	cudaChannelFormatDesc h_colorChannelDesc;
	cudaChannelFormatDesc h_rhoDChannelDesc;
	cudaChannelFormatDesc h_normalChannelDesc;
};