#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h"
#include "CameraTrackingInput.h"

__global__ void renderCorrespondence_kernel(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingParameters cameraTrackingIParameters, float4x4 transform) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const int index1D = x;
	const uint2 index = make_uint2(x % imageWidth, x / imageWidth);

	if (index.x < imageWidth && index.y < imageHeight) {

		float3 pInput = make_float3(cameraTrackingInput.d_inputPos[index1D]);
		float3 nInput = make_float3(cameraTrackingInput.d_inputNormal[index1D]);
		float iInput = cameraTrackingInput.d_inputIntensity[index1D];

		output[x] = MINF;

		if (pInput.x != MINF && !nInput.x != MINF && iInput != MINF )
		{

			//mat3x3 I = intrinsics;
			float4 pInputTransformed = transform * make_float4(pInput, 1);
			float3 pProjTrans = intrinsics * make_float3(pInputTransformed);

			if (pProjTrans.z > 0.f) {

				float2 uvModel = dehomogenize( pProjTrans );
				float3 pTarget = make_float3( getValueNearestNeighbour(uvModel.x, uvModel.y, cameraTrackingInput.d_targetPos, imageWidth, imageHeight) );

				float3 iTargetAndDerivative = make_float3( bilinearInterpolationFloat4( uvModel.x, uvModel.y, cameraTrackingInput.d_targetIntensityAndDerivatives, imageWidth, imageHeight ));

				float iTarget = iTargetAndDerivative.x; // Intensity, uv gradient

				output[x] = iTarget; 
			}
		}
	}
}

extern "C" void renderCorrespondenceCUDA(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingInput cameraTrackingInput, float* intrinsics, CameraTrackingParameters cameraTrackingIParameters, float* transform) {

	const int threadPerBlock = 64;

//	dim3  block(threadPerBlock, threadPerBlock);
//	dim3 grid((imageWidth + threadPerBlock - 1) / threadPerBlock, (imageHeight + threadPerBlock - 1) / threadPerBlock);

	dim3 block(threadPerBlock);
	dim3 grid((imageWidth * imageHeight + threadPerBlock - 1) / threadPerBlock);

	renderCorrespondence_kernel << < grid, block >> > (imageWidth, imageHeight, output, cameraTrackingInput, float3x3(intrinsics), cameraTrackingIParameters, float4x4(transform) );

}