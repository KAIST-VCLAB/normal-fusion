#include "stdafx.h"

#include "VoxelUtilHashSDF.h"

#include "Util.h"

#include "CUDARayCastSDF.h"


extern "C" void renderCS(
	const HashData& hashData,
	const RayCastData &rayCastData, 
	const DepthCameraData &cameraData,
	const RayCastParams &rayCastParams);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void computeNormalsandShading(float4* d_output_normal, float4* d_output_shading, float4* d_input, float4* d_input_rhoD, float4* d_input_detailNormals, float* d_light, unsigned int width, unsigned int height);
extern "C" void computeUniformShading(float4* d_output_shading, float4* d_input, float d_input_rhoD, float* d_light, unsigned int width, unsigned int height);
extern "C" void computeDetailShading(float4* d_output_shading, float4* d_input_normals, float4* d_input_rhoD, float* d_light, unsigned int width, unsigned int height);
extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height, const DepthCameraData& depthCameraData);
extern "C" void computeNormalsShadingVoxel(float4* d_output, float4* d_colorOutput, float4* d_input, float4* d_rhoD, float *light, unsigned int width, unsigned int height);
extern "C" void computeWorldNormals(float4* d_output, float4* d_input, unsigned int width, unsigned height, float4x4 transform);

extern "C" void resetRayIntervalSplatCUDA(RayCastData& data, const RayCastParams& params);
extern "C" void rayIntervalSplatCUDA(const HashData& hashData, const DepthCameraData& cameraData,
								 const RayCastData &rayCastData, const RayCastParams &rayCastParams);


//texture rendering
extern "C" void texRenderCS(
	const HashData& hashData,
	const TexPoolData &texPoolData,
	const RayCastData &rayCastData,
	const DepthCameraData &cameraData,
	const RayCastParams &rayCastParams);
//double texture rendering
extern "C" void doubleTexRenderCS(
	const HashData& hashData,
	const TexPoolData &texPoolData,
	const RayCastData &rayCastData,
	const DepthCameraData &cameraData,
	const RayCastParams &rayCastParams);

extern "C" void doubleTexRenderPrevCS(
	const HashData& hashData,
	const TexPoolData& texPoolData,
	const RayCastData &rayCastData,
	const DepthCameraData &cameraData,
	const RayCastParams &rayCastParams);

extern "C" void doubleTexGeometryRenderCS(const HashData& hashData, const TexPoolData& texPoolData, const RayCastData &rayCastData, const DepthCameraData &cameraData, const RayCastParams &rayCastParams);

Timer CUDARayCastSDF::m_timer;

void CUDARayCastSDF::create(const RayCastParams& params)
{
	m_params = params;
	m_data.allocate(m_params);
	m_rayIntervalSplatting.OnD3D11CreateDevice(DXUTGetD3D11Device(), params.m_width, params.m_height);
}

void CUDARayCastSDF::destroy(void)
{
	m_data.free();
	m_rayIntervalSplatting.OnD3D11DestroyDevice();
}

void CUDARayCastSDF::render(const HashData& hashData, const HashParams& hashParams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{
	rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize()); 
		m_timer.start();
	}

	renderCS(hashData, m_data, cameraData, m_params);

	//convertToCameraSpace(cameraData);
	if (!m_params.m_useGradients)
	{
		if (GlobalAppState::getInstance().s_geometryOptimization) { // Voxel value contains diffuse albedo
			cudaMemcpy(m_data.d_rhoD, m_data.d_colors, sizeof(float4) * m_params.m_height * m_params.m_width, cudaMemcpyDeviceToDevice);
			computeNormalsShadingVoxel(m_data.d_normals, m_data.d_colors, m_data.d_depth4, m_data.d_rhoD, cameraData.d_lightData, m_params.m_width, m_params.m_height);
			computeWorldNormals(m_data.d_normalsWorld, m_data.d_normals, m_params.m_width, m_params.m_height, m_params.m_viewMatrixInverse);
			//cudaMemcpy(m_data.d_colors, m_data.d_rhoD, sizeof(float4) * m_params.m_height * m_params.m_width, cudaMemcpyDeviceToDevice);
		}
		else { // Voxel contains color value
			computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
			computeWorldNormals(m_data.d_normalsWorld, m_data.d_normals, m_params.m_width, m_params.m_height, m_params.m_viewMatrixInverse);
		}
	}

	m_rayIntervalSplatting.unmapCuda();

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize()); 
		m_timer.stop();
		TimingLog::totalTimeRayCast+=m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;
	}
}

void CUDARayCastSDF::convertToCameraSpace(const DepthCameraData& cameraData)
{
	convertDepthFloatToCameraSpaceFloat4(m_data.d_depth4, m_data.d_depth, m_params.m_intrinsicsInverse, m_params.m_width, m_params.m_height, cameraData);
	
	if(!m_params.m_useGradients) {
		
		(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
	}
}

void CUDARayCastSDF::rayIntervalSplatting(const HashData& hashData, const HashParams& hashParams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{
	if (hashParams.m_numOccupiedBlocks == 0)	return;

	if (m_params.m_maxNumVertices <= 6*hashParams.m_numOccupiedBlocks) { // 6 verts (2 triangles) per block
		MLIB_EXCEPTION("not enough space for vertex buffer for ray interval splatting");
	}

	m_params.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_params.m_viewMatrix = MatrixConversion::toCUDA(lastRigidTransform.getInverse());
	m_params.m_viewMatrixInverse = MatrixConversion::toCUDA(lastRigidTransform);

	m_data.updateParams(m_params); 

	//don't use ray interval splatting (cf CUDARayCastSDF.cu -> line 40
	//m_rayIntervalSplatting.rayIntervalSplatting(DXUTGetD3D11DeviceContext(), hashData, cameraData, m_data, m_params, m_params.m_numOccupiedSDFBlocks*6);
}

/////////////////////////////////////////////////////////////////////////
//texture rendering
/////////////////////////////////////////////////////////////////////////

void CUDARayCastSDF::texRender(const HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolparams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{

	rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	// Start query for timing
	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.start();
	}

	texRenderCS(hashData, texPoolData, m_data, cameraData, m_params);

	if (!m_params.m_useGradients) {
		computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
	}

	m_rayIntervalSplatting.unmapCuda();

	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{

		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.stop();
		TimingLog::totalTimeRayCast += m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;

	}
}

/////////////////////////////////////////////////////////////////////////
//double texture rendering
/////////////////////////////////////////////////////////////////////////

void CUDARayCastSDF::setLastViewMatrix(const mat4f& lastRigidTransform) {
	m_params.m_viewMatrix = MatrixConversion::toCUDA(lastRigidTransform.getInverse());
	m_params.m_viewMatrixInverse = MatrixConversion::toCUDA(lastRigidTransform);
}

void CUDARayCastSDF::doubleTexRender(const HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolparams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{
	// Start query for timing
	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.start();
	}

	rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	doubleTexRenderCS(hashData, texPoolData, m_data, cameraData, m_params);

	// TODO: optimize the depth4 and depth using detailNormals, when we needed.

	if (!m_params.m_useGradients) {
		computeNormalsandShading(m_data.d_normals, m_data.d_colors, m_data.d_depth4, m_data.d_rhoD, m_data.d_detailNormals, cameraData.d_lightData, m_params.m_width, m_params.m_height);
		computeWorldNormals(m_data.d_normalsWorld, m_data.d_normals, m_params.m_width, m_params.m_height, m_params.m_viewMatrixInverse);
		computeUniformShading(m_data.d_normalShading, m_data.d_normals, 0.2f, cameraData.d_lightData, m_params.m_width, m_params.m_height);
		computeUniformShading(m_data.d_detailNormalShading, m_data.d_detailNormals, 0.2f, cameraData.d_lightData, m_params.m_width, m_params.m_height);
	}

	cudaMemcpyToArray(m_data.d_rhoDModelArray, 0, 0, m_data.d_rhoD, sizeof(float4)*m_params.m_width*m_params.m_height, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(m_data.d_normalModelArray, 0, 0, m_data.d_detailNormals, sizeof(float4)*m_params.m_width*m_params.m_height, cudaMemcpyDeviceToDevice);


	//cutilSafeCall(cudaMemcpy(m_data.d_normals, m_data.d_detailNormals, sizeof(float4) * m_params.m_height * m_params.m_width, cudaMemcpyDeviceToDevice));
	m_rayIntervalSplatting.unmapCuda();
	
	
	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.stop();
		TimingLog::totalTimeRayCast += m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;

	}
}
void CUDARayCastSDF::doubleTexGeometryRender(const HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolparams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{

	rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	// Start query for timing
	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.start();
	}

	doubleTexGeometryRenderCS(hashData, texPoolData, m_data, cameraData, m_params);

	// TODO: optimize the depth4 and depth using detailNormals, when we needed.

	if (!m_params.m_useGradients) {
		computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
	}

	//cutilSafeCall(cudaMemcpy(m_data.d_normals, m_data.d_detailNormals, sizeof(float4) * m_params.m_height * m_params.m_width, cudaMemcpyDeviceToDevice));
	m_rayIntervalSplatting.unmapCuda();

	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{

		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.stop();
		TimingLog::totalTimeRayCast += m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;

	}
}

void CUDARayCastSDF::doubleTexRenderPrev(const HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolparams, const DepthCameraData& cameraData, const mat4f& lastRigidTransform)
{
	rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	// Start query for timing
	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.start();
	}

	doubleTexRenderPrevCS(hashData, texPoolData, m_data, cameraData, m_params);

	m_rayIntervalSplatting.unmapCuda();

	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{

		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.stop();
		TimingLog::totalTimeRayCast += m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;

	}
}
