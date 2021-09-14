#include "stdafx.h"

#include "CUDABuildLinearSystemEnhanceInput.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include "MatrixConversion.h"

#include <iostream>

extern "C" void computeLightEquations(unsigned int imageWidth, unsigned int imageHeight, float* output, InputEnhanceData m_data, InputEnhanceParams m_params, unsigned int localWindowSize, unsigned int blockSizeInt);


CUDABuildLinearSystemEnhanceInput::CUDABuildLinearSystemEnhanceInput(unsigned int imageWidth, unsigned int imageHeight)
{
	cutilSafeCall(cudaMalloc(&d_output, 57 * sizeof(float)*imageWidth*imageHeight));
	h_output = new float[57 * imageWidth*imageHeight];
}

CUDABuildLinearSystemEnhanceInput::~CUDABuildLinearSystemEnhanceInput() {
	if (d_output) {
		cutilSafeCall(cudaFree(d_output));
	}
	if (h_output) {
		SAFE_DELETE_ARRAY(h_output);
	}
}

void CUDABuildLinearSystemEnhanceInput::applyBL(InputEnhanceData m_data, InputEnhanceParams m_params, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix9x10f& res, LinearSystemConfidence& conf)
{
	unsigned int localWindowSize = 12;
	if (level != 0) localWindowSize = max(1, localWindowSize / (4 * level));

	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)imageWidth*imageHeight) / (localWindowSize*blockSize));

	computeLightEquations(imageWidth, imageHeight, d_output, m_data, m_params, localWindowSize, blockSize);

	cutilSafeCall(cudaMemcpy(h_output, d_output, sizeof(float) * 57 * dimX, cudaMemcpyDeviceToHost));

	// Copy to CPU
	res = reductionSystemCPU(h_output, m_params.weightLightTemp, m_data.h_srcLight, dimX, conf);
}

Matrix9x10f CUDABuildLinearSystemEnhanceInput::reductionSystemCPU(const float* data, const float reg, const float* prev, unsigned int nElems, LinearSystemConfidence& conf)
{
	Matrix9x10f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for (unsigned int k = 0; k < nElems; k++)
	{
		unsigned int linRowStart = 0;

		for (unsigned int i = 0; i < 9; i++)
		{
			for (unsigned int j = i; j < 9; j++)
			{
				res(i, j) += data[57 * k + linRowStart + j - i];
			}

			linRowStart += 9 - i;

			res(i, 9) += data[57 * k + 45 + i];
		}

		conf.sumRegError += data[57 * k + 54];
		conf.sumRegWeight += data[57 * k + 55];

		numCorrF += data[57 * k + 56];
	}

	// Fill lower triangle
	for (unsigned int i = 0; i < 9; i++)
	{
		for (unsigned int j = i; j < 9; j++)
		{
			res(j, i) = res(i, j);
			if (i == j) {
				res(i, j) += reg;
				res(i, 9) += reg * prev[i];
			}
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}
