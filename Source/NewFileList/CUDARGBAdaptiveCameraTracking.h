#pragma once

//#include "stdafx.h"

//#include "Eigen.h"

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "CameraTrackingInput.h"


struct SolverStateCT {

	float *d_masksrc;
	//float *d_masktar;
	
	// PCG iteration
	float *d_F;
	//	float *d_indices_per_pixel;
	float *d_Jp;
	
	float3 *d_Ap_rot;
	float3 *d_Ap_trans;
	float3 *d_p_rot;
	float3 *d_p_trans;
	float3 *d_r_rot;
	float3 *d_r_trans;
	float3 *d_z_rot;
	float3 *d_z_trans;
	float3 *d_precond_rot;
	float3 *d_precond_trans;

	float3 *d_xDelta_rot;
	float3 *d_xDelta_trans;
	float3 *d_x0_rot;
	float3 *d_x0_trans;
	float3 *d_xStep_rot;
	float3 *d_xStep_trans;

	float *d_rDotzOld;

	// our method
	float *d_scanAlpha;
	float *d_scanResidual;

	float3x3 intrinsics;

	float lambda_reg;
	float learning_rate;

	//bool regMode;
	//bool smoothMode;

};

class  CUDARGBAdaptiveCameraTracking {
public:
	CUDARGBAdaptiveCameraTracking(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeW, unsigned int nodeH, unsigned int localWindowWidth);
	~CUDARGBAdaptiveCameraTracking();

	//
	void solve(CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 *d_xDelta_rot, float3 *d_xDelta_trans, float3 *d_xOld_rot, float3 *d_xOld_trans, unsigned int maxInnerIter, unsigned int level);

private:
	SolverStateCT m_solverState;
};