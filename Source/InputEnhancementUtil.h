#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

#include "CUDAInputEnhancementParams.h"

#ifndef __CUDACC__
#include "mLib.h"
#endif

extern __constant__ InputEnhanceParams c_inputEnhanceParams;
extern "C" void updateConstantInputEnhanceParams(const InputEnhanceParams& params);


struct InputEnhanceData
{
	__device__ __host__ 
		InputEnhanceData() {
		d_srcDepth = NULL;
		d_srcAlbedo = NULL;
		d_srcLightMask = NULL;
		d_prevAlbedo = NULL;
		d_updateDepth = NULL;
		d_updateAlbedo = NULL;
		d_updateLight = NULL;

		d_enhanceDepth = NULL;
		d_enhanceAlbedo = NULL;
		d_enhanceLight = NULL;

		d_srcMask = NULL;
		d_albedoTempMask = NULL;
		d_optimizeMask = NULL;

		// PCG Evaluation
		d_sumResidual = NULL;

		// PCG iteration
		d_Jp = NULL;
		d_Jp_shading = NULL;
		d_Jp_reg = NULL;
		d_Jp_smooth = NULL;
		d_Jp_temp = NULL;
		d_Jp_albedo_temp = NULL;
		d_Jp_albedo_spatial = NULL;

		d_Ap_Depth = NULL;
		d_pDepth = NULL;
		d_rDepth = NULL;
		d_zDepth = NULL;
		d_deltaDepth = NULL;
		d_preconditionerDepth = NULL;

		d_Ap_Albedo = NULL;
		d_pAlbedo = NULL;
		d_rAlbedo = NULL;
		d_zAlbedo = NULL;
		d_deltaAlbedo = NULL;
		d_preconditionerAlbedo = NULL;

		d_deltaLight = NULL;

		// Debug variable
		d_debugResidualData = NULL;

		d_rDotzOld = NULL;

		d_scanAlpha = NULL;
		d_scanResidual = NULL;

		// Host variable
		h_intensity = NULL;
		h_mask = NULL;
		h_JtJLight = NULL;
		h_rLight = NULL;
		h_pLight = NULL;
		h_srcLight = NULL;
		h_updateLight = NULL;
		h_preconditionerLight = NULL;
	}

#ifndef __CUDACC__

	void allocate(const InputEnhanceParams& params) {
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcAlbedo, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_prevAlbedo, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcImage, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcLightMask, sizeof(bool) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityImage, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputIntensity, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputNormal, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcLight, sizeof(float) * params.nLightCoefficient));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_updateDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_updateAlbedo, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_updateLight, sizeof(float) * params.nLightCoefficient));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_enhanceDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_enhanceAlbedo, sizeof(float4) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_enhanceLight, sizeof(float) * params.nLightCoefficient));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_sumResidual, sizeof(float) * 8));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp, sizeof(float3) * params.nImagePixel * 2));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_shading, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_reg, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_smooth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_temp, sizeof(float) * 3 * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_albedo_temp, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Jp_albedo_spatial, sizeof(float3) * 4 * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Ap_Depth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_pDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_zDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_deltaDepth, sizeof(float) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_preconditionerDepth, sizeof(float) * params.nImagePixel));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_Ap_Albedo, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_pAlbedo, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rAlbedo, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_zAlbedo, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_deltaAlbedo, sizeof(float3) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_preconditionerAlbedo, sizeof(float3) * params.nImagePixel));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_deltaLight, sizeof(float) * params.nLightCoefficient));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_srcMask, sizeof(bool) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_albedoTempMask, sizeof(bool) * params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_optimizeMask, sizeof(bool) * params.nImagePixel));

		h_intensity = (float *)malloc(sizeof(float) * params.nImagePixel);
		h_mask = (bool *)malloc(sizeof(bool) * params.nImagePixel);

		h_preconditionerLight = (float *)malloc(sizeof(float) * params.nLightCoefficient);
		h_JtJLight = (float *)malloc(sizeof(float) * params.nLightCoefficient * params.nLightCoefficient);
		h_rLight = (float *)malloc(sizeof(float) * params.nLightCoefficient);
		h_pLight = (float *)malloc(sizeof(float) * params.nLightCoefficient);
		h_srcLight = (float *)malloc(sizeof(float) * params.nLightCoefficient);
		h_updateLight = (float *)malloc(sizeof(float) * params.nLightCoefficient);

		// Debug variable
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_debugResidualData, sizeof(float) * params.nImagePixel));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rDotzOld, sizeof(float) *  params.nImagePixel));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_scanAlpha, sizeof(float) * 2));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_scanResidual, sizeof(float))) 
	}

	__host__
		void updateParams(const InputEnhanceParams& params) {
		updateConstantInputEnhanceParams(params);
	}

	__host__
		void free() {
		MLIB_CUDA_SAFE_FREE(d_srcDepth);
		MLIB_CUDA_SAFE_FREE(d_srcLightMask);
		MLIB_CUDA_SAFE_FREE(d_updateDepth);
		MLIB_CUDA_SAFE_FREE(d_enhanceDepth);
		MLIB_CUDA_SAFE_FREE(d_srcAlbedo);
		MLIB_CUDA_SAFE_FREE(d_prevAlbedo);
		MLIB_CUDA_SAFE_FREE(d_updateAlbedo);
		MLIB_CUDA_SAFE_FREE(d_enhanceAlbedo);
		MLIB_CUDA_SAFE_FREE(d_srcImage);
		MLIB_CUDA_SAFE_FREE(d_intensityImage);
		MLIB_CUDA_SAFE_FREE(d_inputIntensity);
		MLIB_CUDA_SAFE_FREE(d_inputNormal);
		MLIB_CUDA_SAFE_FREE(d_srcLight);
		MLIB_CUDA_SAFE_FREE(d_updateLight);
		MLIB_CUDA_SAFE_FREE(d_enhanceLight);
		MLIB_CUDA_SAFE_FREE(d_srcMask);
		MLIB_CUDA_SAFE_FREE(d_albedoTempMask);
		MLIB_CUDA_SAFE_FREE(d_optimizeMask);

		MLIB_CUDA_SAFE_FREE(d_Jp);
		MLIB_CUDA_SAFE_FREE(d_Jp_shading);
		MLIB_CUDA_SAFE_FREE(d_Jp_reg);
		MLIB_CUDA_SAFE_FREE(d_Jp_smooth);
		MLIB_CUDA_SAFE_FREE(d_Jp_temp);
		MLIB_CUDA_SAFE_FREE(d_Jp_albedo_temp);
		MLIB_CUDA_SAFE_FREE(d_Jp_albedo_spatial);

		MLIB_CUDA_SAFE_FREE(d_Ap_Depth);
		MLIB_CUDA_SAFE_FREE(d_pDepth);
		MLIB_CUDA_SAFE_FREE(d_rDepth);
		MLIB_CUDA_SAFE_FREE(d_zDepth);
		MLIB_CUDA_SAFE_FREE(d_deltaDepth);
		MLIB_CUDA_SAFE_FREE(d_preconditionerDepth);

		MLIB_CUDA_SAFE_FREE(d_Ap_Albedo);
		MLIB_CUDA_SAFE_FREE(d_pAlbedo);
		MLIB_CUDA_SAFE_FREE(d_rAlbedo);
		MLIB_CUDA_SAFE_FREE(d_zAlbedo);
		MLIB_CUDA_SAFE_FREE(d_deltaAlbedo);
		MLIB_CUDA_SAFE_FREE(d_preconditionerAlbedo);

		MLIB_CUDA_SAFE_FREE(d_deltaLight);

		MLIB_CUDA_SAFE_FREE(d_sumResidual);
		MLIB_CUDA_SAFE_FREE(d_debugResidualData);

		delete(h_intensity);
		delete(h_mask);
		delete (h_JtJLight);
		delete (h_rLight);
		delete (h_pLight);
		delete (h_srcLight);
		delete (h_updateLight);
		delete (h_preconditionerLight);
	}

#endif

#ifdef __CUDACC__


#endif
	float*		d_srcDepth;
	bool*		d_srcLightMask;
	float4*		d_srcAlbedo;
	float4*		d_prevAlbedo;
	bool*		d_srcMask;
	bool*		d_albedoTempMask;
	bool*		d_optimizeMask;
	float*		d_srcLight;

	float*		d_updateDepth;
	float*		d_enhanceDepth;

	float4*		d_updateAlbedo;
	float4*		d_enhanceAlbedo;

	float*		d_updateLight;
	float*		d_enhanceLight;

	float4*		d_srcImage;
	float*		d_intensityImage;		
	float*		d_inputIntensity;
	float4*		d_inputNormal;

	// PCG Evaluation
	float *d_sumResidual;

	// PCG iteration
	float3 *d_Jp;
	float3 *d_Jp_shading;
	float *d_Jp_reg;
	float *d_Jp_smooth;
	float *d_Jp_temp;
	float3 *d_Jp_albedo_temp;
	float3 *d_Jp_albedo_spatial;

	float *d_Ap_Depth;
	float *d_pDepth;
	float *d_rDepth;
	float *d_zDepth;
	float *d_deltaDepth;
	float *d_preconditionerDepth;

	float3 *d_Ap_Albedo;
	float3 *d_pAlbedo;
	float3 *d_rAlbedo;
	float3 *d_zAlbedo;
	float3 *d_deltaAlbedo;
	float3 *d_preconditionerAlbedo;

	float *h_intensity;
	bool *h_mask;

	float *d_deltaLight;

	// Debug variable
	float *d_debugResidualData;

	float *d_rDotzOld;

	// our method
	float *d_scanAlpha;
	float *d_scanResidual;

	float *h_JtJLight;
	float *h_rLight;
	float *h_pLight;
	float *h_srcLight;
	float *h_updateLight;
	float *h_preconditionerLight;
};