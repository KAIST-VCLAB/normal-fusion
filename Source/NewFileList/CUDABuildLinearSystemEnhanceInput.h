#pragma once

/************************************************************************/
/* Linear System Build on the GPU for ICP                               */
/************************************************************************/

#include "stdafx.h"

#include "Eigen.h"
#include "ICPErrorLog.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include "InputEnhancementUtil.h"


class CUDABuildLinearSystemEnhanceInput
{
public:

	CUDABuildLinearSystemEnhanceInput(unsigned int imageWidth, unsigned int imageHeight);
	~CUDABuildLinearSystemEnhanceInput();

	void applyBL(InputEnhanceData m_data, InputEnhanceParams m_params, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix9x10f& res, LinearSystemConfidence& conf);

	//! builds AtA, AtB, and confidences
	Matrix9x10f CUDABuildLinearSystemEnhanceInput::reductionSystemCPU(const float* data, const float reg, const float* prev, unsigned int nElems, LinearSystemConfidence& conf);

private:

	float* d_output;
	float* h_output;
};
