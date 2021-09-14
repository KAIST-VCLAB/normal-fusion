#pragma once

#include "stdafx.h"

#include "Eigen.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include "CameraTrackingInput.h"
#include "ICPErrorLog.h"


class CUDABuildLinearSystemLocalDepth
{
public:

	CUDABuildLinearSystemLocalDepth(unsigned int imageWidth, unsigned int imageHeight);
	CUDABuildLinearSystemLocalDepth(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeWidth, unsigned int nodeHeight);
	~CUDABuildLinearSystemLocalDepth();

	void applyBLs(CameraTrackingDepthLocalInput cameraTrackingInput, Eigen::Matrix3f & intrinsics, CameraTrackingDepthLocalParameters cameraTrackingParameters, float3 * d_x_rot, float3 * d_x_trans, float3 * d_x_step_rot, float3 * d_x_step_trans, float lambdaReg, float * d_system, unsigned int level, LinearSystemConfidence & conf);
	

	//! builds AtA, AtB, and confidences
	Matrix6x7f reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf);

private:

	float* d_output;
	float *d_temp;
	float* h_output;
};
