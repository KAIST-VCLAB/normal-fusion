#include "stdafx.h"

#include "CUDABuildLinearSystemLocalDepth.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include "MatrixConversion.h"

#include <iostream>


#define BLOCK_SIZE 256

extern "C" void computeNormalEquationsAllRegDepthLocal(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambda, CameraTrackingDepthLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingDepthLocalParameters cameraTrackingParameters);


CUDABuildLinearSystemLocalDepth::CUDABuildLinearSystemLocalDepth(unsigned int imageWidth, unsigned int imageHeight)
{
	cutilSafeCall(cudaMalloc(&d_output, 30 * sizeof(float)*imageWidth*imageHeight));
	h_output = new float[30 * imageWidth*imageHeight];
}

CUDABuildLinearSystemLocalDepth::CUDABuildLinearSystemLocalDepth(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeWidth, unsigned int nodeHeight)
{
//	cutilSafeCall(cudaMalloc(&d_output, 30 * sizeof(float)*imageWidth*imageHeight));
//	h_output = new float[30 * imageWidth*imageHeight];
	cutilSafeCall(cudaMalloc(&d_temp, 30 * BLOCK_SIZE * sizeof(float)* nodeWidth * nodeHeight));

}

CUDABuildLinearSystemLocalDepth::~CUDABuildLinearSystemLocalDepth() {
	if (d_output) {
		cutilSafeCall(cudaFree(d_output));
	}
	if (h_output) {
		SAFE_DELETE_ARRAY(h_output);
	}
}


void CUDABuildLinearSystemLocalDepth::applyBLs(CameraTrackingDepthLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingDepthLocalParameters cameraTrackingParameters, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambdaReg, float *d_system, unsigned int level, LinearSystemConfidence& conf)
{

	Eigen::Matrix3f intrinsicsRowMajor = intrinsics.transpose(); // Eigen is col major / cuda is row major
	computeNormalEquationsAllRegDepthLocal(cameraTrackingParameters.imageWidth, cameraTrackingParameters.imageHeight, d_system, d_temp, d_x_rot, d_x_trans, d_x_step_rot, d_x_step_trans, lambdaReg,cameraTrackingInput, intrinsicsRowMajor.data(), cameraTrackingParameters);
	
}


Matrix6x7f CUDABuildLinearSystemLocalDepth::reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf)
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for (unsigned int k = 0; k<nElems; k++)
	{
		unsigned int linRowStart = 0;

		for (unsigned int i = 0; i<6; i++)
		{
			for (unsigned int j = i; j<6; j++)
			{
				res(i, j) += data[30 * k + linRowStart + j - i];
			}

			linRowStart += 6 - i;

			res(i, 6) += data[30 * k + 21 + i];
		}

		conf.sumRegError += data[30 * k + 27];
		conf.sumRegWeight += data[30 * k + 28];

		numCorrF += data[30 * k + 29];
	}

	// Fill lower triangle
	for (unsigned int i = 0; i<6; i++)
	{
		for (unsigned int j = i; j<6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}
