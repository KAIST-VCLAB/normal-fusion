#include "stdafx.h"

#include "stdafx.h"

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "TexUpdateUtil.h"
#include "InputEnhancementUtil.h"

#include "Util.h"

#include "CUDATexUpdate.h"
#include "CUDAInputEnhancement.h"

#include "cudaDebug.h"

#include "DebugVisualizationTool.h"

#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include <iostream>
#include <limits>

#include "modeDefine.h"

/////////////////////////////////////////////////////
// Input Enhancement Multi Res
/////////////////////////////////////////////////////

Timer CUDAInputEnhancement::m_timer;

extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void resampleFloat3Map(float3* d_colorMapResampledFloat3, unsigned int outputWidth, unsigned int outputHeight, float3* d_colorMapFloat3, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight);

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void solveEnhancementStub(InputEnhanceData& state, InputEnhanceParams& enhanceParams, const RayCastParams& rayCastParams);

extern "C" void initializeEnhanceOptimizerMaps(float4* d_output, bool* d_mask, bool* d_inputMask, unsigned int width, unsigned int height, float value = 1.0f);
extern "C" void computeDepthNormalMask(InputEnhanceData m_data, InputEnhanceParams m_params);
extern "C" void computeOptimizationMask(InputEnhanceData m_data, InputEnhanceParams m_params);
extern "C" void computeAlbedoTempValidMask(InputEnhanceData m_data, InputEnhanceParams m_params);
extern "C" void computeDepthNormals(float4* d_output, float* d_input, unsigned int width, unsigned int height, float fx, float fy, float cx, float cy);
extern "C" void computeCanonicalDepthNormals(float4* d_output, float* d_input, const float4x4& extrinsic, unsigned int width, unsigned int height, float fx, float fy, float cx, float cy);
extern "C" void convertBooleanToFloat(float* d_output, bool* d_input, unsigned int width, unsigned int height);
extern "C" void copyDepthFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void copyDepthFloatMapWithMask(float* d_output, float* d_input, bool* mask, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void computeLightMask(bool* d_output, float* d_input, unsigned int width, unsigned int height, float fx, float fy, float cx, float cy);
extern "C" void copyAlbedoFloatMapWithMask(float4* d_output, float4* d_input, bool* mask, unsigned int width, unsigned int height, float minDepth, float maxDepth);

void CUDAInputEnhancement::create(const InputEnhanceParams& params)
{
	m_params = params;
	m_data.allocate(m_params);

	m_CUDABuildLinearSystem = new CUDABuildLinearSystemEnhanceInput(m_params.imageWidth, m_params.imageHeight);
}

void CUDAInputEnhancement::destroy(void)
{
	m_data.free();
}

void CUDAInputEnhancement::computeMask(void)
{
	computeDepthNormalMask(m_data, m_params);
}

Vector9f CUDAInputEnhancement::computeBestRigidAlignment(unsigned int maxInnerIter, LinearSystemConfidence& conf)
{
	conf.reset();

	Matrix9x10f system;

	m_CUDABuildLinearSystem->applyBL(m_data, m_params, m_params.imageWidth, m_params.imageHeight, m_params.currentLevel, system, conf);

	Eigen::Matrix<float, 9, 9> ATA = system.block(0, 0, 9, 9);
	Eigen::Matrix<float, 9, 1> ATb = system.block(0, 9, 9, 1);

	if (ATA.isZero()) {
		printf("ERROR??\n\n");
		getchar();
	}
	else {
		Eigen::JacobiSVD<Eigen::Matrix<float, 9, 9>> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix<float, 9, 1> x = SVD.solve(ATb);

		//computing the matrix condition
		Eigen::Matrix<float, 9, 1> evs = SVD.singularValues();
		conf.matrixCondition = evs[0] / evs[9];

		if (x(0, 0) != -std::numeric_limits<float>::infinity())
		{
			return x;
		}
		else {
			printf("ERROR??\n\n");
			getchar();
		}
	}
}

Vector9f CUDAInputEnhancement::align(unsigned int maxInnerIter, unsigned maxOuterIter, ICPErrorLog* errorLog)
{
	float lastICPError = -1.0f;
	Vector9f light;
	for (unsigned int i = 0; i < maxOuterIter; i++)
	{
		LinearSystemConfidence currConf;

		light = computeBestRigidAlignment(maxInnerIter, currConf);

		if (errorLog) {
			errorLog->addCurrentICPIteration(currConf, m_params.currentLevel);
		}
	}

	return light;
}


void CUDAInputEnhancement::enhanceInput(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const RayCastData &rayCastData, const RayCastParams& rayCastParams, const TexUpdateData& texUpdateData,
	const std::vector<unsigned int>& maxOuterIter,
	const std::vector<unsigned int>& maxInnerIter,
	const std::vector<float>& weightsDataShading,
	const std::vector<float>& weightsDepthTemp,
	const std::vector<float>& weightsDepthSmooth,
	const std::vector<float>& weightsAlbedoTemp,
	const std::vector<float>& weightsAlbedoSmooth,
	const std::vector<float>& weightsLightTemp,
	bool firstIter,
	int frame)
{
	m_params.originalFx = depthCameraParams.fx;
	m_params.originalFy = depthCameraParams.fy;
	m_params.originalMx = depthCameraParams.mx;
	m_params.originalMy = depthCameraParams.my;

	m_params.originalImageHeight = depthCameraParams.m_imageHeight;
	m_params.originalImageWidth = depthCameraParams.m_imageWidth;
	m_params.originalNumImagePixel = m_params.originalImageHeight * m_params.originalImageWidth;

	m_params.isFirstIter = firstIter;

	computeDepthNormals(depthCameraData.d_inputNormalData, depthCameraData.d_depthData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight, depthCameraParams.fx, depthCameraParams.fy, depthCameraParams.mx, depthCameraParams.my);

	for (int nLevel = m_params.nLevel - 1; nLevel >= 0; nLevel--) {
		// Setting parameter for current level.
		int LevelParameter = pow(2, nLevel);

		m_params.currentLevel = nLevel;
		m_params.levelParams = LevelParameter * LevelParameter * 2 * 2;
		m_params.fx = depthCameraParams.fx / (float)LevelParameter;
		m_params.fy = depthCameraParams.fy / (float)LevelParameter;
		m_params.mx = depthCameraParams.mx / (float)LevelParameter;
		m_params.my = depthCameraParams.my / (float)LevelParameter;

		m_params.imageHeight = depthCameraParams.m_imageHeight / LevelParameter;
		m_params.imageWidth = depthCameraParams.m_imageWidth / LevelParameter;
		m_params.nImagePixel = m_params.imageHeight * m_params.imageWidth;

		m_params.nPCGOuterIteration = maxOuterIter[nLevel];
		m_params.nPCGInnerIteration = maxInnerIter[nLevel];

		m_params.weightDataShading = weightsDataShading[nLevel];
		m_params.weightDepthTemp = weightsDepthTemp[nLevel];
		m_params.weightDepthSmooth = weightsDepthSmooth[nLevel];
		m_params.weightAlbedoTemp = weightsAlbedoTemp[nLevel];
		m_params.weightAlbedoSmooth = weightsAlbedoSmooth[nLevel];
		m_params.weightLightTemp = weightsLightTemp[nLevel];


		resampleFloat4Map(m_data.d_srcImage, m_params.imageWidth, m_params.imageHeight, depthCameraData.d_colorData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
		resampleFloatMap(m_data.d_srcDepth, m_params.imageWidth, m_params.imageHeight, depthCameraData.d_depthData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
		if (!firstIter) resampleFloat4Map(m_data.d_prevAlbedo, m_params.imageWidth, m_params.imageHeight, rayCastData.d_rhoD, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
		convertColorToIntensityFloat(m_data.d_intensityImage, m_data.d_srcImage, m_params.imageWidth, m_params.imageHeight);

		computeDepthNormalMask(m_data, m_params);
		computeOptimizationMask(m_data, m_params);
		if (!firstIter) computeAlbedoTempValidMask(m_data, m_params);

		if (nLevel == m_params.nLevel - 1) {
			if (firstIter) {
				float initialLight[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				cudaMemcpy(m_data.d_srcLight, initialLight, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyHostToDevice);
				memcpy(m_data.h_srcLight, initialLight, sizeof(float) * m_params.nLightCoefficient);
			}
			else {
				cudaMemcpy(m_data.d_srcLight, depthCameraData.d_lightData, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyDeviceToDevice);
				cudaMemcpy(m_data.h_srcLight, m_data.d_srcLight, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(m_data.h_intensity, m_data.d_intensityImage, sizeof(float) * m_params.nImagePixel, cudaMemcpyDeviceToHost);
			cudaMemcpy(m_data.h_mask, m_data.d_srcMask, sizeof(bool) * m_params.nImagePixel, cudaMemcpyDeviceToHost);
			float sum = 0.0f;
			int count = 0;
			for (int idx = 0; idx < m_params.nImagePixel; idx++) {
				if (m_data.h_mask[idx]) {
					sum += m_data.h_intensity[idx];
					count++;
				}
			}
			m_params.initialAlbedo = sum / float(count);

			initializeEnhanceOptimizerMaps(m_data.d_srcAlbedo, m_data.d_srcLightMask, m_data.d_optimizeMask, m_params.imageWidth, m_params.imageHeight, m_params.initialAlbedo);

			cudaMemcpy(m_data.d_updateDepth, m_data.d_srcDepth, sizeof(float) * m_params.nImagePixel, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_data.d_updateAlbedo, m_data.d_srcAlbedo, sizeof(float4) * m_params.nImagePixel, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_data.d_updateLight, m_data.d_srcLight, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyDeviceToDevice);
		}

		if (frame <= 2) {
			m_params.weightAlbedoTemp = 0.0f;
			m_params.weightLightTemp = 0.0f;
		}

		convertColorToIntensityFloat(m_data.d_inputIntensity, m_data.d_updateAlbedo, m_params.imageWidth, m_params.imageHeight);
		computeCanonicalDepthNormals(m_data.d_inputNormal, m_data.d_updateDepth, rayCastParams.m_viewMatrixInverse, m_params.imageWidth, m_params.imageHeight, m_params.fx, m_params.fy, m_params.mx, m_params.my);
		computeLightMask(m_data.d_srcLightMask, m_data.d_updateDepth, m_params.imageWidth, m_params.imageHeight, m_params.fx, m_params.fy, m_params.mx, m_params.my);

		{
			LinearSystemConfidence currConf;
			Vector9f light = computeBestRigidAlignment(1, currConf);
			light = light / light(0) * m_params.ambientLightBase;
			cudaMemcpy(m_data.d_updateLight, light.data(), sizeof(float) * 9, cudaMemcpyHostToDevice);
		}

		if (nLevel == m_params.nLevel - 1 && !firstIter) cudaMemcpy(m_data.d_updateAlbedo, m_data.d_prevAlbedo, sizeof(float4) * m_params.nImagePixel, cudaMemcpyDeviceToDevice);
		
		solveEnhancementStub(m_data, m_params, rayCastParams);

		if (nLevel > 0) {
			resampleFloat4Map(m_data.d_enhanceAlbedo, m_params.imageWidth * 2, m_params.imageHeight * 2, m_data.d_updateAlbedo, m_params.imageWidth, m_params.imageHeight);
			resampleFloatMap(m_data.d_enhanceDepth, m_params.imageWidth * 2, m_params.imageHeight * 2, m_data.d_updateDepth, m_params.imageWidth, m_params.imageHeight);
			cudaMemcpy(m_data.d_updateDepth, m_data.d_enhanceDepth, sizeof(float) * m_params.originalNumImagePixel, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_data.d_updateAlbedo, m_data.d_enhanceAlbedo, sizeof(float4) * m_params.originalNumImagePixel, cudaMemcpyDeviceToDevice);
		}
	}

	if (!GlobalAppState::get().s_geometryOptimization)
	{
		float initialLight2[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		cudaMemcpy(depthCameraData.d_lightData, initialLight2, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyHostToDevice);
		cudaMemcpy(depthCameraData.d_rhoDData, depthCameraData.d_colorData, sizeof(float4)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(depthCameraData.d_rhoDArray, 0, 0, depthCameraData.d_rhoDData, sizeof(float4)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
		computeDepthNormals(depthCameraData.d_normalData, depthCameraData.d_depthData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight, depthCameraParams.fx, depthCameraParams.fy, depthCameraParams.mx, depthCameraParams.my);
		cudaMemcpyToArray(depthCameraData.d_normalArray, 0, 0, depthCameraData.d_normalData, sizeof(float4)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
	}
	else
	{
		if (firstIter) {
			if (!GlobalAppState::get().s_voxelRenderingEnabled) {
				convertBooleanToFloat(texUpdateData.d_depthMask, m_data.d_srcMask, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
			}
			else {
				convertBooleanToFloat(depthCameraData.d_depthMask, m_data.d_srcMask, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
			}
		}
		copyDepthFloatMapWithMask(depthCameraData.d_depthData, m_data.d_updateDepth, m_data.d_srcMask, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight, depthCameraParams.m_sensorDepthWorldMin, depthCameraParams.m_sensorDepthWorldMax);
		cudaMemcpyToArray(depthCameraData.d_depthArray, 0, 0, depthCameraData.d_depthData, sizeof(float)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
		cudaMemcpy(depthCameraData.d_lightData, m_data.d_updateLight, sizeof(float) * m_params.nLightCoefficient, cudaMemcpyDeviceToDevice);
		copyAlbedoFloatMapWithMask(depthCameraData.d_rhoDData, m_data.d_updateAlbedo, m_data.d_srcMask, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight, depthCameraParams.m_sensorDepthWorldMin, depthCameraParams.m_sensorDepthWorldMax);
		cudaMemcpyToArray(depthCameraData.d_rhoDArray, 0, 0, m_data.d_updateAlbedo, sizeof(float4)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
		computeDepthNormals(depthCameraData.d_normalData, depthCameraData.d_depthData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight, depthCameraParams.fx, depthCameraParams.fy, depthCameraParams.mx, depthCameraParams.my);
		cudaMemcpyToArray(depthCameraData.d_normalArray, 0, 0, depthCameraData.d_normalData, sizeof(float4)*depthCameraParams.m_imageHeight*depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
	}

	if (GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{

		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.stop();
		TimingLog::totalTimeEnhanceInput += m_timer.getElapsedTimeMS();
		TimingLog::countTimeEnhanceInput++;

	}
}