#pragma once

#include "GlobalAppState.h"
#include "TimingLog.h"

#include "MatrixConversion.h"
#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"
#include "RayCastSDFUtil.h"

#include "DX11RayIntervalSplatting.h"

class CUDARayCastSDF
{
public:
	CUDARayCastSDF(const RayCastParams& params) {
		create(params);
	}

	~CUDARayCastSDF(void) {
		destroy();
	}

	static RayCastParams parametersFromGlobalAppState(const GlobalAppState& gas, const mat4f& intrinsics, const mat4f& intrinsicsInv) {
		RayCastParams params;
		params.m_width = gas.s_adapterWidth;
		params.m_height = gas.s_adapterHeight;
		params.m_intrinsics = MatrixConversion::toCUDA(intrinsics);
		params.m_intrinsicsInverse = MatrixConversion::toCUDA(intrinsicsInv);
		params.m_minDepth = gas.s_sensorDepthMin;
		params.m_maxDepth = gas.s_sensorDepthMax;
		params.m_rayIncrement = gas.s_SDFRayIncrementFactor * gas.s_SDFTruncation;
		params.m_thresSampleDist = gas.s_SDFRayThresSampleDistFactor * params.m_rayIncrement;
		params.m_thresDist = gas.s_SDFRayThresDistFactor * params.m_rayIncrement;
		params.m_useGradients = gas.s_SDFUseGradients;

		params.m_maxNumVertices = gas.s_hashNumSDFBlocks * 6;

		return params;
	}

	void render(const HashData& hashData, const HashParams& hashParams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform);
	
	void texRender(const HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolParams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform);

	void setLastViewMatrix(const mat4f & lastRigidTransform);

	void doubleTexRender(const HashData & hashData, const HashParams & hashParams, const TexPoolData & texPoolData, const TexPoolParams & texPoolparams, const DepthCameraData & cameraData, const mat4f & lastRigidTransform);

	void doubleTexGeometryRender(const HashData & hashData, const HashParams & hashParams, const TexPoolData & texPoolData, const TexPoolParams & texPoolparams, const DepthCameraData & cameraData, const mat4f & lastRigidTransform);

	void doubleTexRenderPrev(const HashData & hashData, const HashParams & hashParams, const TexPoolData & texPoolData, const TexPoolParams & texPoolparams, const DepthCameraData & cameraData, const mat4f & lastRigidTransform);

	const RayCastData& getRayCastData(void) {
		return m_data;
	}

	const RayCastParams& getRayCastParams() const {
		return m_params;
	}


	// debugging
	void convertToCameraSpace(const DepthCameraData& cameraData);
	void rayIntervalSplatting(const HashData& hashData, const HashParams& hashParams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform); // rasterize

private:

	void create(const RayCastParams& params);
	void destroy(void);


	RayCastParams m_params;
	RayCastData m_data;

	DX11RayIntervalSplatting m_rayIntervalSplatting;

	static Timer m_timer;
};

