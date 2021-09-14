#pragma once

#include "RGBDSensor.h"
#include "GlobalAppState.h"

#include <cuda_runtime.h> 
#include <cuda_d3d11_interop.h> 
#include "cudaUtil.h"
#include "cuda_SimpleMatrixUtil.h"

#include "CUDARGBDAdapter.h"
#include "CUDAScan.h"
#include "CUDACameraTrackingMultiRes.h"
#include "DepthCameraUtil.h"

#include "DX11RGBDRenderer.h"
#include "DX11CustomRenderTarget.h"

#include <cstdlib>

extern "C" void computeDerivativesFloat(float* d_outputDU, float* d_outputDV, float* d_input, unsigned int width, unsigned int height);

class CUDARGBDSensor
{
	public:

		CUDARGBDSensor();
		~CUDARGBDSensor();

		void OnD3D11DestroyDevice();
			
		HRESULT OnD3D11CreateDevice(ID3D11Device* device, CUDARGBDAdapter* CUDARGBDAdapter);

		HRESULT setBlurScore(int numKey, int numPass);

		HRESULT process(ID3D11DeviceContext* context);

		//! enables bilateral filtering of the depth value
		void setFiterDepthValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);
		void setFiterIntensityValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);


		float* getDepthMapRawFloat(){
			return m_RGBDAdapter->getDepthMapResampledFloat();
		}
		float* getDepthMapFilteredFloat() {
			return d_depthMapFilteredFloat;
		}

		float* getDepthMapColorSpaceFloat();
		float4* getCameraSpacePositionsFloat4();
		float4* getColorMapFilteredFloat4();
		float4* getNormalMapFloat4();

		float* getIntensityMapFilteredFloat()
		{
			return d_intensityMapFilteredFloat;
		}

		float* GetIntensityDerivativeDUMapFloat();
		float* GetIntensityDerivativeDVMapFloat();

		unsigned int getDepthWidth() const;
		unsigned int getDepthHeight() const;
		unsigned int getColorWidth() const;
		unsigned int getColorHeight() const;
		
		bool checkValidT1(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4);
		bool checkValidT2(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4);

		//! the depth camera data (lives on the GPU)
		const DepthCameraData& getDepthCameraData() {
			return m_depthCameraData;
		}

		//! the depth camera parameter struct (lives on the CPU)
		const DepthCameraParams& getDepthCameraParams() {
			return m_depthCameraParams;
		}

		mat4f getRigidTransform() const {
			return m_RGBDAdapter->getRigidTransform();
		}


		//! computes and returns the depth map in hsv
		float4* getAndComputeDepthHSV() const;
		
		bool passBlurScore(int framecnt) const {
			if (!GlobalAppState::get().s_blurnessRejectionEnable || framecnt <= numKeyScoreCounter + 1) return true;
			float score = blurScores[(scoreCounter + 1) % maxScoreCounter];
			int count = 0;
			for (int i = scoreCounter + 1; i < scoreCounter + 1 + numKeyScoreCounter; i++) {
				int index = i % maxScoreCounter;
				if (score > blurScores[index]) count++;
			}
			return (count <= numPassScore);
		}

	private:
		void appendScore(float score) {
			blurScores[scoreCounter--] = score;
			if (scoreCounter < 0) scoreCounter = maxScoreCounter - 1;
		}

		DepthCameraData		m_depthCameraData;
		DepthCameraParams	m_depthCameraParams;
	
		CUDARGBDAdapter* m_RGBDAdapter;

		DX11RGBDRenderer			g_RGBDRenderer;
		DX11CustomRenderTarget		g_CustomRenderTarget;

		bool  m_bFilterDepthValues;
		float m_fBilateralFilterSigmaD;
		float m_fBilateralFilterSigmaR;

		bool  m_bFilterIntensityValues;
		float m_fBilateralFilterSigmaDIntensity;
		float m_fBilateralFilterSigmaRIntensity;

		//! depth texture float [D]
		float*			d_depthMapFilteredFloat;

		//! camera space pos float [X Y Z *]
		float4*			d_cameraSpaceFloat4;

		//! normal texture float [X Y Z *]
		float4*	d_normalMapFloat4;

		//! intensity
		float* d_intensityMapFilteredFloat;

		//! tmp erode helpers
		float*	d_depthErodeHelper;
		float4*	d_colorErodeHelper;

		//! hsv depth for visualization
		float4* d_depthHSV;

		// For the image blurness
		float *b_ver, *b_hor;
		float *d_f_ver, *d_b_ver;
		float *d_f_hor, *d_b_hor;
		float *v_ver, *v_hor;
		float *d_sum;

		float* blurScores;
		int scoreCounter;
		int numPassScore;
		unsigned int numKeyScoreCounter;
		unsigned int maxScoreCounter;

		Timer m_timer;
};
