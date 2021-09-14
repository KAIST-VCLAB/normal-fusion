#include "cudaUtil.h"

#pragma once


struct CameraTrackingParameters
{
	 float weightDepth;
	 float weightColor;
	 float distThres;
	 float normalThres;

	 float sensorMaxDepth;
	 float colorGradiantMin;
	 float colorThres;
};

struct CameraTrackingInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float*  d_inputIntensity;
	float4* d_targetPos;
	float4* d_targetNormal;
	float4* d_targetIntensityAndDerivatives;
};


struct CameraTrackingLocalParameters
{
	float lambdaReg;
	float lambdaNormals;
	float lambdaColors;
	int localWindowHWidth;

	float sensorMaxDepth;
	float colorGradiantMin;
	float colorThres;

	float2 offset;
	float2 cellWH;
	float sigma;
	int nodeWidth, nodeHeight;
	int imageWidth, imageHeight;
};

struct CameraTrackingLocalInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float*  d_inputIntensity;
	float4* d_inputIntensity4;
	float* d_inputMask;
	float* d_targetMask;
	float4* d_targetIntensityAndDerivatives;

	float4* d_targetIntensityAndDerivativesX;
	float4* d_targetIntensityAndDerivativesY;
	float4* d_targetIntensityAndDerivativesZ;
	float4* d_targetIntensityAndDerivativesW;
};

struct CameraTrackingDepthLocalParameters
{
	float distThres;
	float normalThres;

	float weightColor;
	float colorGradiantMin;
	float colorThres;

	float lambdaReg;
	int localWindowHWidth;

	float sensorMaxDepth;

	float2 offset;
	float2 cellWH;
	float sigma;
	int nodeWidth, nodeHeight;
	int imageWidth, imageHeight;
};

struct CameraTrackingDepthLocalInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float* d_inputIntensity;
	float4* d_targetPos;
	float4* d_targetNormal;
	float4* d_targetIntensityAndDerivatives;

	float* d_inputMask;
	float* d_targetMask;
};