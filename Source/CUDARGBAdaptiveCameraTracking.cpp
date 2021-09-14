#include "stdafx.h"

#include "CUDARGBAdaptiveCameraTracking.h"
#include "GlobalAppState.h"
#include "cudaDebug.h"
//#include "MatrixConversion.h"

extern"C" void PCGInit(CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingLocalParameters,  SolverStateCT solverState);
extern"C" void PCGProcess(CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingLocalParameters, SolverStateCT solverState, bool lastIteration);

CUDARGBAdaptiveCameraTracking::CUDARGBAdaptiveCameraTracking(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeW, unsigned int nodeH, unsigned int localWindowHWidth) {

	const int pixelN = imageWidth * imageHeight;
	const int nodeN = nodeW * nodeH;
	const int localWindowWidth = localWindowHWidth * 2 + 1;
	const int localWindowN = localWindowWidth * localWindowWidth;

	//cudaMalloc(&m_solverState.d_F, sizeof(float) * nodeN * localWindowWidth * localWindowWidth);
	cudaMalloc(&m_solverState.d_Jp, sizeof(float) * nodeN * localWindowN);
	cudaMalloc(&m_solverState.d_Ap_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_Ap_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_p_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_p_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_r_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_r_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_xDelta_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_xDelta_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_z_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_z_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_precond_rot, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_precond_trans, sizeof(float3) * nodeN);
	cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float) * nodeN);

	cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float) * 2);
	cudaMalloc(&m_solverState.d_scanResidual, sizeof(float) * 3);

}

CUDARGBAdaptiveCameraTracking::~CUDARGBAdaptiveCameraTracking() {

	if (m_solverState.d_Jp)		cudaFree(m_solverState.d_Jp);
	if (m_solverState.d_Ap_rot)		cudaFree(m_solverState.d_Ap_rot);
	if (m_solverState.d_Ap_trans)		cudaFree(m_solverState.d_Ap_trans);
	if (m_solverState.d_p_rot)		cudaFree(m_solverState.d_p_rot);
	if (m_solverState.d_p_trans)		cudaFree(m_solverState.d_p_trans);
	if (m_solverState.d_r_rot)		cudaFree(m_solverState.d_r_rot);
	if (m_solverState.d_r_trans)		cudaFree(m_solverState.d_r_trans);
	if (m_solverState.d_xDelta_rot)		cudaFree(m_solverState.d_xDelta_rot);
	if (m_solverState.d_xDelta_trans)		cudaFree(m_solverState.d_xDelta_trans);
	if (m_solverState.d_z_rot)		cudaFree(m_solverState.d_Jp);
	if (m_solverState.d_z_trans)		cudaFree(m_solverState.d_z_rot);
	if (m_solverState.d_precond_rot)		cudaFree(m_solverState.d_precond_rot);
	if (m_solverState.d_precond_trans)		cudaFree(m_solverState.d_precond_trans);
	if (m_solverState.d_rDotzOld)		cudaFree(m_solverState.d_rDotzOld);
	if (m_solverState.d_scanAlpha)		cudaFree(m_solverState.d_scanAlpha);
	if (m_solverState.d_scanResidual)		cudaFree(m_solverState.d_scanResidual);
}

void  CUDARGBAdaptiveCameraTracking::solve(CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 *d_xStep_rot, float3 *d_xStep_trans, float3 *d_xOld_rot, float3 *d_xOld_trans, unsigned int maxInnerIter, unsigned int level) {

	//m_solverState.
	//d_xDelta, d_xOld
	//imageWidth, imageHeight, level
	m_solverState.d_xStep_rot = d_xStep_rot;
	m_solverState.d_xStep_trans = d_xStep_trans;
	m_solverState.d_x0_rot = d_xOld_rot;
	m_solverState.d_x0_trans = d_xOld_trans;

	m_solverState.intrinsics = float3x3 ( intrinsics );

	m_solverState.lambda_reg = 0.1;
	m_solverState.learning_rate = 1;
	PCGInit( cameraTrackingInput, cameraTrackingParameters, m_solverState);
	
	bool lastIteration = false;

	for (int pcgIt = 0; pcgIt < maxInnerIter; pcgIt++) {
		printf("pcgIt %d\n", pcgIt);
		if (pcgIt == maxInnerIter- 1)
			lastIteration = true;

		PCGProcess(cameraTrackingInput, cameraTrackingParameters, m_solverState, lastIteration);

	}

}