#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h"
#include "InputEnhancementUtil.h"

/////////////////////////////////////////////////////
// Defines
/////////////////////////////////////////////////////

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 57
#endif

#define MINF __int_as_float(0xff800000)

/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////

__shared__ float bucket7[ARRAY_SIZE*BLOCK_SIZE];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////

__device__ inline void addToLocalScanElement(uint inpGTid, uint resGTid, volatile float* shared)
{
#pragma unroll
	for (uint i = 0; i < ARRAY_SIZE; i++)
	{
		shared[ARRAY_SIZE*resGTid + i] += shared[ARRAY_SIZE*inpGTid + i];
	}
}

__device__ inline void CopyToResultScanElement(uint GID, float* output)
{
#pragma unroll
	for (uint i = 0; i < ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID + i] = bucket7[0 + i];
	}
}

__device__ inline void SetZeroScanElement(uint GTid)
{
#pragma unroll
	for (uint i = 0; i < ARRAY_SIZE; i++)
	{
		bucket7[GTid*ARRAY_SIZE + i] = 0.0f;
	}
}


struct Float1x9
{
	float data[9];
};
/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket7);
	addToLocalScanElement(GTid + 16, GTid, bucket7);
	addToLocalScanElement(GTid + 8, GTid, bucket7);
	addToLocalScanElement(GTid + 4, GTid, bucket7);
	addToLocalScanElement(GTid + 2, GTid, bucket7);
	addToLocalScanElement(GTid + 1, GTid, bucket7);
}

/////////////////////////////////////////////////////
// Compute Normal Equations
/////////////////////////////////////////////////////

__device__ inline  void addToLocalSystem(mat1x9& ABlockRow, mat1x1& residualsBlockRow, float weight, uint threadIdx, volatile float* shared)
{
	uint linRowStart = 0;

#pragma unroll
	for (uint i = 0; i < 9; i++)
	{
		mat1x1 colI; ABlockRow.getBlock(0, i, colI);

#pragma unroll
		for (uint j = i; j < 9; j++)
		{
			mat1x1 colJ; ABlockRow.getBlock(0, j, colJ);

			shared[ARRAY_SIZE*threadIdx + linRowStart + j - i] += colI.getTranspose()*colJ*weight; // ATA
		}

		linRowStart += 9 - i;

		shared[ARRAY_SIZE*threadIdx + 45 + i] += colI.getTranspose()*residualsBlockRow*weight; // ATb
	}

	shared[ARRAY_SIZE*threadIdx + 54] += weight * residualsBlockRow.norm1DSquared(); // residual
	shared[ARRAY_SIZE*threadIdx + 55] += weight;									 // weight

	shared[ARRAY_SIZE*threadIdx + 56] += 1.0f;									 // corr number
}

__global__ void scanLightEquationsDevice(unsigned int imageWidth, unsigned int imageHeight, float* output, InputEnhanceData m_data, InputEnhanceParams m_params, unsigned int localWindowSize)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	// Set system to zero
	SetZeroScanElement(threadIdx.x);

	//Locally sum small window
#pragma unroll
	for (uint i = 0; i < localWindowSize; i++)
	{
		const int index1D = localWindowSize * x + i;
		const uint2 index = make_uint2(index1D%imageWidth, index1D / imageWidth);

		if (index.x < imageWidth && index.y < imageHeight)
		{
			mat3x1 nInput = mat3x1(make_float3(m_data.d_inputNormal[index1D]));
			mat1x1 iInput = mat1x1(m_data.d_inputIntensity[index1D]);
			mat1x1 iTarget = mat1x1(m_data.d_intensityImage[index1D]);

			if (!nInput.checkMINF() && !iInput.checkMINF() && !iTarget.checkMINF() && m_data.d_srcLightMask[index1D])
			{
				mat1x9 ABlockRow;

				ABlockRow(0) = 1.0f;
				ABlockRow(1) = nInput(1);
				ABlockRow(2) = nInput(2);
				ABlockRow(3) = nInput(0);
				ABlockRow(4) = nInput(0) * nInput(1);
				ABlockRow(5) = nInput(1) * nInput(2);
				ABlockRow(6) = (-nInput(0) * nInput(0) - nInput(1) * nInput(1) + 2.f * nInput(2) * nInput(2));
				ABlockRow(7) = nInput(2) * nInput(0);
				ABlockRow(8) = (nInput(0) * nInput(0) - nInput(1) * nInput(1));

				ABlockRow = iInput * ABlockRow;
				addToLocalSystem(ABlockRow, iTarget, m_params.weightDataShading, threadIdx.x, bucket7);
			}
		}
	}

	__syncthreads();

	// Up sweep 2D
#pragma unroll
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElement(threadIdx.x + stride / 2, threadIdx.x, bucket7);

		__syncthreads();
	}

	if (threadIdx.x < 32) warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElement(blockIdx.x, output);
}

extern "C" void computeLightEquations(unsigned int imageWidth, unsigned int imageHeight, float* output, InputEnhanceData m_data, InputEnhanceParams m_params, unsigned int localWindowSize, unsigned int blockSizeInt)
{
	const unsigned int numElements = imageWidth * imageHeight;
	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt * localWindowSize - 1) / (blockSizeInt*localWindowSize), 1, 1);

	scanLightEquationsDevice << <gridSize, blockSize >> > (imageWidth, imageHeight, output, m_data, m_params, localWindowSize);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}