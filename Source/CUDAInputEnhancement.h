#pragma once
#include "GlobalAppState.h"
#include "GlobalEnhancementState.h"
#include "TimingLog.h"
#include "ICPErrorLog.h"
#include "Eigen.h"

#include "MatrixConversion.h"
#include "CUDABuildLinearSystemEnhanceInput.h"

#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"
#include "InputEnhancementUtil.h"
#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"

#include "cudaDebug.h"


class CUDAInputEnhancement {

public:

	CUDAInputEnhancement(const InputEnhanceParams& params) {
		create(params);
	}

	~CUDAInputEnhancement(void) {
		destroy();
	}

	CUDAInputEnhancement() {

	}

	Vector9f CUDAInputEnhancement::computeBestRigidAlignment(unsigned int maxInnerIter, LinearSystemConfidence& conf);
	Vector9f align(unsigned int maxInnerIter, unsigned maxOuterIter, ICPErrorLog * errorLog);

	void enhanceInput(HashData & hashData, const HashParams & hashParams, const DepthCameraData & depthCameraData, const DepthCameraParams & depthCameraParams, const RayCastData & rayCastData, const RayCastParams & rayCastParams, const TexUpdateData & texUpdateData, const std::vector<unsigned int>& maxOuterIter, const std::vector<unsigned int>& maxInnerIter, const std::vector<float>& weightsDataShading, const std::vector<float>& weightsDepthTemp, const std::vector<float>& weightsDepthSmooth, const std::vector<float>& weightsAlbedoTemp, const std::vector<float>& weightsAlbedoSmooth, const std::vector<float>& weightsLightTemp, bool firstIter, int frame);

	static InputEnhanceParams parametersFromGlobalState(const GlobalAppState& gas, const GlobalEnhancementState& ges, const mat4f& intrinsics) {
		InputEnhanceParams params;

		params.fx = intrinsics(0, 0);
		params.fy = intrinsics(1, 1);
		params.mx = intrinsics(0, 2);
		params.my = intrinsics(1, 2);

		params.imageWidth = gas.s_adapterWidth;
		params.imageHeight = gas.s_adapterHeight;
		params.nImagePixel = params.imageWidth * params.imageHeight;

		params.originalFx = intrinsics(0, 0);
		params.originalFy = intrinsics(1, 1);
		params.originalMx = intrinsics(0, 2);
		params.originalMy = intrinsics(1, 2);

		params.originalImageWidth = gas.s_adapterWidth;
		params.originalImageHeight = gas.s_adapterHeight;
		params.originalNumImagePixel = params.originalImageWidth * params.originalImageHeight;

		params.nLightCoefficient = gas.s_numLightCoefficients;

		params.optimizeAlbedo = ges.s_optimizeAlbedo;
		params.optimizeLight = ges.s_optimizeLight;
		params.optimizeDepth = ges.s_optimizeDepth;
		params.optimizeFull = ges.s_optimizeFull;

		params.weightDataShading = ges.s_weightsDataShading[0];
		params.weightDepthTemp = ges.s_weightsDepthTemp[0];
		params.weightDepthSmooth = ges.s_weightsDepthSmooth[0];
		params.weightAlbedoSmooth = ges.s_weightsAlbedoSmooth[0];
		params.ambientLightBase = ges.s_ambientLightBase;

		params.nPCGInnerIteration = ges.s_maxOuterIter[0];
		params.nPCGOuterIteration = ges.s_maxInnerIter[0];
		params.nLevel = ges.s_maxLevels;

		params.currentLevel = 0;
		params.levelParams = 1;
		params.initialAlbedo = ges.s_initialAlbedo;
		return params;
	}

private:
	void create(const InputEnhanceParams& params);
	void destroy(void);
	void computeMask(void);

	InputEnhanceData m_data;
	InputEnhanceParams m_params;
	
	CUDABuildLinearSystemEnhanceInput* m_CUDABuildLinearSystem;

	static Timer m_timer;
};
