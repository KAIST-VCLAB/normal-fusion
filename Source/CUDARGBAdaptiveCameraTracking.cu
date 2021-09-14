#include "CUDARGBAdaptiveCameraTracking.h"
#include "cudaDebug.h"
#include "ICPUtil.h"

#define THREAD_PER_BLOCK 32
#define THREAD_PER_BLOCK_JT 64
#define WARP_SIZE 32
#define FLOAT_EPSILON 0.000001f
////////////////////////////////////////////////////////////////////////
cudaEvent_t g_startct, g_stopct;
bool timerctEnabled = false;

void cudaTimerStartct() {
	if (timerctEnabled)
		cudaEventRecord(g_startct);
}

void cudaTimerStopct(std::string a) {
	if (timerctEnabled) {
		float milliseconds;
		cudaEventRecord(g_stopct);
		cudaEventSynchronize(g_stopct);
		cudaEventElapsedTime(&milliseconds, g_startct, g_stopct);
		printf("%s: %f\n", a.data(), milliseconds);
	}
}

////////////////////////////////////////////////////////////////////////

__device__ inline float weightDist(float2 src, float2 tar, float sigma) {
	float2 diff = src - tar;

	return exp(-(diff.x * diff.x + diff.y * diff.y) / (2 * sigma*sigma));

}

////////////////////////////////////////////////////////////////////////

__inline__ __device__ float warpReduce(float val) {
	int offset = 32 >> 1;
	while (offset > 0) {
		val = val + __shfl_down(val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}

__inline__ __device__ float warpReduce(float3 &val) {
	val.x = warpReduce(val.x);
	val.y = warpReduce(val.y);
	val.z = warpReduce(val.z);
}

////////////////////////////////////////////////////////////////////////

__inline__ __device__ void evalJtF(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters, int lane, int nodeInd, int threadInd, float3& rRot, float3& rTrans, float3& pRot, float3& pTrans) {
	//compute the position of nodes

	rRot = make_float3(0.f, 0.f, 0.f);
	rTrans = make_float3(0.f, 0.f, 0.f);
	pRot = make_float3(0.f, 0.f, 0.f);
	pTrans = make_float3(0.f, 0.f, 0.f);

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeIndX = nodeInd % nodeW;
	const int nodeIndY = nodeInd / nodeW;
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;

	float2 nodePos = cellWH * make_float2(nodeIndX, nodeIndY)+cameraTrackingParameters.offset;
	int2 nodePosi = make_int2(nodePos);

	int six = max(0, nodePosi.x - localWindowHWidth) ;
	int eix = min(imageW - 1, nodePosi.x + localWindowHWidth+1);
	int siy = max(0, nodePosi.y - localWindowHWidth);
	int eiy = min(imageH - 1, nodePosi.y + localWindowHWidth+1);

	int rangeW = eix - six; // sx <= x < ex
	int rangeH = eiy - siy; // sy <= y < ey
	int rangeN = rangeW * rangeH;

	for (int i = threadInd; i < rangeN; i += THREAD_PER_BLOCK_JT) {

		int offix = i % rangeW;
		int offiy = i / rangeW;
		int ix = six + offix;
		int iy = siy + offiy;
		int index1D = iy * imageW + ix;

		mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
		mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));
		mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);
		//float maskVal = cameraTrackingInput.d_inputMask[index1D];

		if ( !pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF() && iInput > 1.f / 255.f)
		{
			//
			float3 x0Rot = solverState.d_x0_rot[nodeInd];
			float3 x0Trans = solverState.d_x0_trans[nodeInd];
			float3 xStepRot = solverState.d_xStep_rot[nodeInd];
			float3 xStepTrans = solverState.d_xStep_trans[nodeInd];

			float3 curRot = x0Rot + xStepRot;
			float3 curTrans = x0Trans + xStepTrans;

			float3 aRot, aTrans;

			mat3x3 I = mat3x3(solverState.intrinsics);
			mat3x3 ROld = mat3x3(evalRMat(curRot));
			mat3x1 pInputTransformed = ROld*pInput + mat3x1(curTrans);
			mat3x1 nInputTransformed = ROld*nInput;

			mat3x3 Ralpha = mat3x3(evalR_dGamma(curRot));
			mat3x3 Rbeta = mat3x3(evalR_dBeta(curRot));
			mat3x3 Rgamma = mat3x3(evalR_dAlpha(curRot));

			mat3x1 pProjTrans = I*pInputTransformed;

			if (pProjTrans(2) > 0.0f)
			{
				mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

				mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageW, imageH)));
				mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

				mat2x3 PI = dehomogenizeDerivative(pProjTrans);
				mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

				if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
				{

					// Color
					mat1x1 diffIntensity(iTarget - iInput);
					mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);


					if (DIntensity.norm1D() > cameraTrackingParameters.colorGradiantMin)
					{
						float weightColor;
						//= max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingParameters.colorThres);

						weightColor = weightDist(make_float2(ix, iy), nodePos, cameraTrackingParameters.sigma);
						mat1x3 tmp0Intensity = DIntensity*PI*I;

						aRot = -weightColor * make_float3(tmp0Intensity*phiAlpha, tmp0Intensity*phiBeta, tmp0Intensity*phiGamma);
						aTrans = -weightColor * make_float3(tmp0Intensity(0), tmp0Intensity(1), tmp0Intensity(2));

						rRot += aRot * diffIntensity(0, 0);// .norm1D();// (0, 0);
						rTrans += aTrans * diffIntensity(0, 0);// .norm1D();
						pRot += aRot * aRot;
						pTrans += aTrans * aTrans;

					}
				}
			}
		}
	}


	warpReduce(rRot);
	warpReduce(rTrans);
	warpReduce(pRot);
	warpReduce(pTrans);
}

__inline__ __device__ float applyJp(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters, int pixelInd, float2 pixelPos, int nodeInd, float2 nodePos ) {


	
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;

	float res = 0.f;

	mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[pixelInd]));
	mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[pixelInd]));
	mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[pixelInd]);
	//float maskVal = cameraTrackingInput.d_inputMask[pixelInd];

	if ( !pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF() && iInput > 1.f / 255.f)
	{
		//
		float3 x0Rot = solverState.d_x0_rot[nodeInd];
		float3 x0Trans = solverState.d_x0_trans[nodeInd];
		float3 xStepRot = solverState.d_xStep_rot[nodeInd];
		float3 xStepTrans = solverState.d_xStep_trans[nodeInd];
		float3 pRot = solverState.d_p_rot[nodeInd];
		float3 pTrans = solverState.d_p_trans[nodeInd];


		float3 curRot = x0Rot + xStepRot;
		float3 curTrans = x0Trans + xStepTrans;

		float3 aRot, aTrans;

		mat3x3 I = mat3x3(solverState.intrinsics);
		mat3x3 ROld = mat3x3(evalRMat(curRot));
		mat3x1 pInputTransformed = ROld*pInput + mat3x1(curTrans);
		mat3x1 nInputTransformed = ROld*nInput;

		mat3x3 Ralpha = mat3x3(evalR_dGamma(curRot));
		mat3x3 Rbeta = mat3x3(evalR_dBeta(curRot));
		mat3x3 Rgamma = mat3x3(evalR_dAlpha(curRot));

		mat3x1 pProjTrans = I*pInputTransformed;

		if (pProjTrans(2) > 0.0f)
		{
			mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

			mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageW, imageH)));
			mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

			mat2x3 PI = dehomogenizeDerivative(pProjTrans);
			mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

			if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
			{

				// Color
				mat1x1 diffIntensity(iTarget - iInput);
				mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);


				if (DIntensity.norm1D() > cameraTrackingParameters.colorGradiantMin)
				{
					float weightColor;
					//= max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingParameters.colorThres);

					weightColor = weightDist(pixelPos, nodePos, cameraTrackingParameters.sigma);
					mat1x3 tmp0Intensity = DIntensity*PI*I;

					aRot = weightColor * make_float3(tmp0Intensity*phiAlpha, tmp0Intensity*phiBeta, tmp0Intensity*phiGamma);
					aTrans = weightColor * make_float3(tmp0Intensity(0), tmp0Intensity(1), tmp0Intensity(2));

					res = (dot(aRot, pRot) + dot(aTrans,pTrans));


				}
			}
		}

	}

	return res;
}

__global__ void PCG_step_data_kernel0(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeN = nodeW * nodeH;

	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;
	
	const int localWindowWidth = localWindowHWidth * 2 + 1;
	const int localPixelN = localWindowWidth * localWindowWidth;

	const int resN = nodeN * localPixelN;

	const int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
	
	const int nodeInd = threadInd / localPixelN;
	const int pixInd = threadInd % localPixelN;
	
	const int nodeIndX = nodeInd % nodeW;
	const int nodeIndY = nodeInd / nodeW;
	const int pixIndX = pixInd % localWindowWidth;
	const int pixIndY = pixInd / localWindowWidth;
	
	float2 nodePos = cellWH * make_float2(nodeIndX, nodeIndY)+cameraTrackingParameters.offset;
	int2 nodePosi = make_int2(nodePos);

	int2 pixelPosi = make_int2 ( nodePosi.x - localWindowHWidth + pixIndX, nodePosi.y - localWindowHWidth + pixIndY );

	if (threadInd < resN) {

		solverState.d_Jp[threadInd] = 0;

		if (0 <= pixelPosi.x  && pixelPosi.x < imageW && 0 <= pixelPosi.y && pixelPosi.y < imageH) {

			int pixelInd = pixelPosi.x + pixelPosi.y * imageW;
			float tmp;

			tmp = applyJp(solverState, cameraTrackingInput, cameraTrackingParameters, pixelInd, make_float2(pixelPosi), nodeInd, nodePos );
			solverState.d_Jp[threadInd] = tmp;

		}
	}
}

__inline__ __device__ void applyJtJp(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters,  int lane, int nodeInd, int threadInd, float3& rRot, float3& rTrans) {
	rRot = make_float3(0.f, 0.f, 0.f);
	rTrans = make_float3(0.f, 0.f, 0.f);

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeIndX = nodeInd % nodeW;
	const int nodeIndY = nodeInd / nodeW;
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const int localWindowWidth = 2 * localWindowHWidth  + 1;
	const int localWindowN = localWindowWidth * localWindowWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;

	float2 nodePos = cellWH * make_float2(nodeIndX, nodeIndY)+ cameraTrackingParameters.offset;
	int2 nodePosi = make_int2(nodePos);

	int six =  nodePosi.x - localWindowHWidth;
	int eix = min(imageW - 1, nodePosi.x + localWindowHWidth+1);
	int siy = nodePosi.y - localWindowHWidth;
	int eiy = min(imageH - 1, nodePosi.y + localWindowHWidth+1);

	for (int i = threadInd; i < localWindowN; i += THREAD_PER_BLOCK_JT) {

		int offix = i % localWindowWidth;
		int offiy = i / localWindowWidth;

		int JPInd = nodeInd * localWindowN + i;

		int ix = six + offix;
		int iy = siy + offiy;


		if (0 <= ix && ix < imageW && 0 <= iy && iy < imageH) {

			int index1D = iy * imageW + ix;

			mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
			mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));
			mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);
			float maskVal = cameraTrackingInput.d_inputMask[index1D];

			if (!pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF() && iInput > 1.f / 255.f)
			{
				//
				float3 x0Rot = solverState.d_x0_rot[nodeInd];
				float3 x0Trans = solverState.d_x0_trans[nodeInd];
				float3 xStepRot = solverState.d_xStep_rot[nodeInd];
				float3 xStepTrans = solverState.d_xStep_trans[nodeInd];

				float3 curRot = x0Rot + xStepRot;
				float3 curTrans = x0Trans + xStepTrans;

				float3 aRot, aTrans;

				mat3x3 I = mat3x3(solverState.intrinsics);
				mat3x3 ROld = mat3x3(evalRMat(curRot));
				mat3x1 pInputTransformed = ROld*pInput + mat3x1(curTrans);
				mat3x1 nInputTransformed = ROld*nInput;

				mat3x3 Ralpha = mat3x3(evalR_dGamma(curRot));
				mat3x3 Rbeta = mat3x3(evalR_dBeta(curRot));
				mat3x3 Rgamma = mat3x3(evalR_dAlpha(curRot));

				mat3x1 pProjTrans = I*pInputTransformed;

				if (pProjTrans(2) > 0.0f)
				{
					mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

					mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageW, imageH)));
					mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

					mat2x3 PI = dehomogenizeDerivative(pProjTrans);
					mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

					if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
					{

						// Color
						mat1x1 diffIntensity(iTarget - iInput);
						mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);


						if (DIntensity.norm1D() > cameraTrackingParameters.colorGradiantMin)
						{
							float weightColor;
							//= max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingParameters.colorThres);

							weightColor = weightDist(make_float2(ix, iy), nodePos, cameraTrackingParameters.sigma);
							mat1x3 tmp0Intensity = DIntensity*PI*I;

							aRot = weightColor * make_float3(tmp0Intensity*phiAlpha, tmp0Intensity*phiBeta, tmp0Intensity*phiGamma);
							aTrans = weightColor * make_float3(tmp0Intensity(0), tmp0Intensity(1), tmp0Intensity(2));

							rRot += aRot * solverState.d_Jp[JPInd];
							rTrans += aTrans * solverState.d_Jp[JPInd];

						}
					}
				}
			}
		}
	}

	warpReduce(rRot);
	warpReduce(rTrans);
}

__global__ void PCG_step_data_kernel1(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeN = cameraTrackingParameters.nodeHeight *  cameraTrackingParameters.nodeWidth;
	const unsigned int nodeInd = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;
	
	if (0 <= nodeInd && nodeInd < nodeN) {

		float3 resRot, resTrans;
		
		resRot = make_float3(0.f, 0.f, 0.f);
		resTrans = make_float3(0.f, 0.f, 0.f);
		//__shared__ float2 prerot = make_float2(0.f, 0.f);


		applyJtJp(solverState, cameraTrackingInput, cameraTrackingParameters, lane, nodeInd, threadIdx.x, resRot, resTrans );
		//evalJtF(solverState, cameraTrackingInput, cameraTrackingParameters, lane, nodeInd, threadIdx.x, rRot, rTrans, pRot, pTrans);

		__syncthreads();

		if (lane == 0) {

			//check sign
			atomicAdd(&solverState.d_Ap_rot[nodeInd].x, resRot.x);
			atomicAdd(&solverState.d_Ap_rot[nodeInd].y, resRot.y);
			atomicAdd(&solverState.d_Ap_rot[nodeInd].z, resRot.z);
			atomicAdd(&solverState.d_Ap_trans[nodeInd].x, resTrans.x);
			atomicAdd(&solverState.d_Ap_trans[nodeInd].y, resTrans.y);
			atomicAdd(&solverState.d_Ap_trans[nodeInd].z, resTrans.z);

		}
	}
}

__global__ void PCG_step_data_kernel2(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeN = cameraTrackingParameters.nodeHeight *  cameraTrackingParameters.nodeWidth;
	const unsigned int nodeInd = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;

	if (0 <= nodeInd && nodeInd < nodeN)
	{
		//	solverState.d_Ap_flow[nodeInd] += solverState.lambda_reg * solverState.lambda_reg *solverState.d_p_flow[nodeInd];
		//compute Reg
		solverState.d_Ap_rot[nodeInd] += solverState.lambda_reg * solverState.d_p_rot[nodeInd];
		solverState.d_Ap_trans[nodeInd] += solverState.lambda_reg * solverState.d_p_trans[nodeInd];
		d = dot(solverState.d_p_rot[nodeInd], solverState.d_Ap_rot[nodeInd]) + dot(solverState.d_p_trans[nodeInd], solverState.d_Ap_trans[nodeInd]);		// x-th term of denominator of alpha
	}

	d = warpReduce(d);

	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(solverState.d_scanAlpha, d);
	}
}

__global__ void PCG_step_data_kernel3(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const unsigned int nodeN = cameraTrackingParameters.nodeWidth * cameraTrackingParameters.nodeHeight;
	const unsigned int nodeInd = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotproduct = solverState.d_scanAlpha[0];
	float b = 0.f;

	if (0 <= nodeInd && nodeInd < nodeN ) {
		float alpha = 0.0f;
		if (dotproduct > FLOAT_EPSILON) alpha = solverState.d_rDotzOld[nodeInd] / dotproduct;		// update step size alpha

		//printf("alpha: %f dotproduct: %f dotold: %f \n", alpha, dotproduct, solverState.d_rDotzOld[nodeInd]);
		solverState.d_xDelta_rot[nodeInd] = solverState.d_xDelta_rot[nodeInd] +  alpha*solverState.d_p_rot[nodeInd];			// do a decent step
		solverState.d_xDelta_trans[nodeInd] = solverState.d_xDelta_trans[nodeInd] + alpha*solverState.d_p_trans[nodeInd];			// do a decent step


		float3 rRot = solverState.d_r_rot[nodeInd] - alpha*solverState.d_Ap_rot[nodeInd];					// update residuum
		float3 rTrans = solverState.d_r_trans[nodeInd] - alpha*solverState.d_Ap_trans[nodeInd];					// update residuum
		solverState.d_r_rot[nodeInd] = rRot;														// store for next kernel call
		solverState.d_r_trans[nodeInd] = rTrans;														// store for next kernel call

		float3 zRot = rRot / solverState.d_precond_rot[nodeInd];														// apply preconditioner M^-1
		float3 zTrans = rTrans / solverState.d_precond_trans[nodeInd];														// apply preconditioner M^-1
		solverState.d_z_rot[nodeInd] = zRot;	// save for next kernel call
		solverState.d_z_trans[nodeInd] = zTrans;	// save for next kernel call

		b = dot(zRot, rRot) + dot(zTrans, rTrans);									// compute x-th term of the nominator of beta
	}

	b = warpReduce(b);

	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(&solverState.d_scanAlpha[1], b);
	}
}

__global__ void PCG_step_data_kernel4(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters, bool lastIteration) {

	const unsigned int nodeN = cameraTrackingParameters.nodeWidth * cameraTrackingParameters.nodeHeight;
	const unsigned int nodeInd = blockIdx.x * blockDim.x + threadIdx.x;

		if (0 <= nodeInd && nodeInd < nodeN)
	{
		const float rDotzNew = solverState.d_scanAlpha[1];								// get new nominator
		const float rDotzOld = solverState.d_rDotzOld[nodeInd];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;				// update step size beta

		solverState.d_rDotzOld[nodeInd] = rDotzNew;											// save new rDotz for next iteration
		// In 2013 CGF they use r, but PCG use z. z is correct.
		solverState.d_p_rot[nodeInd] = solverState.d_z_rot[nodeInd] + beta * solverState.d_p_rot[nodeInd];		// update decent direction
		solverState.d_p_trans[nodeInd] = solverState.d_z_trans[nodeInd] + beta * solverState.d_p_trans[nodeInd];		// update decent direction
		solverState.d_Ap_rot[nodeInd] = make_float3(0.0f, 0.0f, 0.0f);
		solverState.d_Ap_trans[nodeInd] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{

			//if (input.d_validImages[x]) { //not really necessary
			float sign = 1.f;
			solverState.d_xStep_rot[nodeInd] = solverState.d_xStep_rot[nodeInd] + solverState.learning_rate *sign * solverState.d_xDelta_rot[nodeInd];
			solverState.d_xStep_trans[nodeInd] = solverState.d_xStep_trans[nodeInd] + solverState.learning_rate * sign * solverState.d_xDelta_trans[nodeInd];
			//solverState.d_x_trans[nodeInd] = solverState.d_x_trans[nodeInd] + solverState.d_delta_trans[nodeInd];
			//}
		}
	}
}

__global__ void PCG_initialize_data_kernel0(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeN = cameraTrackingParameters.nodeHeight *  cameraTrackingParameters.nodeWidth;
	const unsigned int nodeInd = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	if (0 <= nodeInd && nodeInd < nodeN) {

		float3 rRot, rTrans;
		float3 pRot, pTrans;
		__shared__ float3 resRot, resTrans;
		__shared__ float3 preRot, preTrans;
		resRot = make_float3(0.f, 0.f, 0.f);
		resTrans = make_float3(0.f, 0.f, 0.f);
		preRot = make_float3(0.f, 0.f, 0.f);
		preTrans = make_float3(0.f, 0.f, 0.f);
		//__shared__ float2 prerot = make_float2(0.f, 0.f);

		evalJtF(solverState, cameraTrackingInput, cameraTrackingParameters, lane, nodeInd, threadIdx.x, rRot, rTrans, pRot, pTrans);

		__syncthreads();

		if (lane == 0) {

			//check sign
			atomicAdd(&resRot.x, rRot.x);
			atomicAdd(&resRot.y, rRot.y);
			atomicAdd(&resRot.z, rRot.z);
			atomicAdd(&resTrans.x, rTrans.x);
			atomicAdd(&resTrans.y, rTrans.y);
			atomicAdd(&resTrans.z, rTrans.z);
			atomicAdd(&preRot.x, pRot.x);
			atomicAdd(&preRot.y, pRot.y);
			atomicAdd(&preRot.z, pRot.z);
			atomicAdd(&preTrans.x, pTrans.x);
			atomicAdd(&preTrans.y, pTrans.y);
			atomicAdd(&preTrans.z, pTrans.z);

		}

		__syncthreads();

		if (threadIdx.x == 0) {

			solverState.d_r_rot[nodeInd] = resRot;
			solverState.d_r_trans[nodeInd] = resTrans;
			solverState.d_precond_rot[nodeInd] = preRot;
			solverState.d_precond_trans[nodeInd] = preTrans;

		}
	}
}

__global__ void PCG_initialize_reg_kernel0(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {


	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeN = nodeW * nodeH;
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;

	const unsigned int nodeInd  = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 <= nodeInd && nodeInd < nodeN) {

		//float norm = sqrt (dot(solverState.d_node_motion[nodeInd], solverState.d_node_motion[nodeInd]));
		//solverState.d_r_rot[nodeInd] -= solverState.lambda_reg * solverState.d_xStep_rot[nodeInd];
		//solverState.d_r_trans[nodeInd] -= solverState.lambda_reg * solverState.d_xStep_trans[nodeInd];
		solverState.d_precond_rot[nodeInd] += solverState.lambda_reg * make_float3(1, 1, 1);
		solverState.d_precond_trans[nodeInd] += solverState.lambda_reg *  make_float3(1, 1, 1);
		solverState.d_p_rot[nodeInd] += solverState.d_r_rot[nodeInd] / solverState.d_precond_rot[nodeInd];
		solverState.d_p_trans[nodeInd] += solverState.d_r_trans[nodeInd] / solverState.d_precond_trans[nodeInd];

		float d = dot(solverState.d_r_trans[nodeInd], solverState.d_p_trans[nodeInd])+ dot(solverState.d_r_rot[nodeInd], solverState.d_p_rot[nodeInd]);

		solverState.d_Ap_rot[nodeInd] = make_float3(0.f, 0.f, 0.f);
		solverState.d_Ap_trans[nodeInd] = make_float3(0.f, 0.f, 0.f);

		d = warpReduce(d);
		
		if (threadIdx.x % WARP_SIZE == 0)
			atomicAdd(solverState.d_scanAlpha, d);

	}
}

__global__ void PCG_initialize_final_kernel(SolverStateCT solverState, CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeN = nodeW * nodeH;
	const unsigned int nodeInd = blockIdx.x * blockDim.x + threadIdx.x;
	float d = 0.0f;

	if (0 <= nodeInd && nodeInd < nodeN)
		solverState.d_rDotzOld[nodeInd] = solverState.d_scanAlpha[0];

}

extern"C" void PCGInit(CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters, SolverStateCT solverState) {
	
	const unsigned int nodeN = cameraTrackingParameters.nodeHeight * cameraTrackingParameters.nodeWidth;
	const unsigned int localWindowWidth = cameraTrackingParameters.localWindowHWidth * 2 + 1;
	const unsigned int resN = nodeN * localWindowWidth* localWindowWidth;

	const int blockNodeN = (nodeN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	//const int blockResN = (resN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	
	cudaError hr = cudaSuccess;

	bool debug = false;

	if (debug) {
		hr = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(hr, "PCG_initialization", "beforeStart");
	}

	//reset xDelta
	cudaMemset(solverState.d_xDelta_rot, 0, sizeof(float3) * nodeN);
	cudaMemset(solverState.d_xDelta_trans, 0, sizeof(float3) * nodeN);

	cudaMemset(solverState.d_scanAlpha, 0, sizeof(float) * 2);
	cudaMemset(solverState.d_scanResidual, 0, sizeof(float) * 2);

	PCG_initialize_data_kernel0 << < nodeN, THREAD_PER_BLOCK_JT >> > (solverState, cameraTrackingInput, cameraTrackingParameters);

	//writeArray<float>((float*)solverState.d_r_rot, "1d_r_rot.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_r_trans, "1d_r_trans.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_precond_rot, "1d_precond_rot.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_precond_trans, "1d_precond_trans.txt", nodeN,  3);

	PCG_initialize_reg_kernel0 << < blockNodeN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters);

	//writeArray<float>((float*)solverState.d_Ap_rot, "2d_Ap_rot.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_Ap_trans, "2d_Ap_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_p_rot, "2d_p_rot.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_p_trans, "2d_p_trans.txt", nodeN,  3);
	//writeArray<float>((float*)solverState.d_r_rot, "2d_r_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_r_trans, "2d_r_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_precond_rot, "2d_precond_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_precond_trans, "2d_precond_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_scanAlpha, "2d_scanAlpha.txt", 1, 3);


	PCG_initialize_final_kernel << < blockNodeN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters);
	//writeArray<float>((float*)solverState.d_rDotzOld, "3d_rDotzOld.txt", nodeN, 1);
}
extern"C" void PCGProcess(CameraTrackingLocalInput cameraTrackingInput, CameraTrackingLocalParameters cameraTrackingParameters, SolverStateCT solverState, bool lastIteration) {

	const unsigned int nodeN = cameraTrackingParameters.nodeHeight * cameraTrackingParameters.nodeWidth;
	const unsigned int localWindowWidth = cameraTrackingParameters.localWindowHWidth * 2 + 1;
	const unsigned int resN = nodeN * localWindowWidth* localWindowWidth;
	
	const int blockResN = (resN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	const int blockNodeN = (nodeN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	PCG_step_data_kernel0 << < blockResN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters);
//	writeArray<float>((float*)solverState.d_Jp, "p0d_Jp.txt",resN, 1);

	PCG_step_data_kernel1 << < nodeN, THREAD_PER_BLOCK_JT >> > (solverState, cameraTrackingInput, cameraTrackingParameters);
	//writeArray<float>((float*)solverState.d_Ap_trans, "p1d_Ap_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_Ap_rot, "p1d_Ap_rots.txt", nodeN, 3);

	cudaMemset(solverState.d_scanAlpha, 0, sizeof(float) * 2);
	PCG_step_data_kernel2 << < blockNodeN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters);

	//writeArray<float>((float*)solverState.d_Ap_trans, "p2d_Ap_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_Ap_rot, "p2d_Ap_rots.txt", nodeN, 3);

	PCG_step_data_kernel3 << < blockNodeN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters);
	//writeArray<float>((float*)solverState.d_xDelta_rot, "p3d_xDelta_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_xDelta_trans, "p3d_xDelta_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_r_rot, "p3d_r_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_r_trans, "p3d_r_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_z_rot, "p3d_z_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_z_trans, "p3d_z_trans.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_scanAlpha, "p3d_scanAlpha.txt", 3, 1);

	cudaMemset(solverState.d_scanResidual, 0, sizeof(float) * 3);
	PCG_step_data_kernel4 << < blockNodeN, THREAD_PER_BLOCK >> > (solverState, cameraTrackingInput, cameraTrackingParameters, lastIteration);
	//writeArray<float>((float*)solverState.d_p_rot, "p4d_p_rot.txt", nodeN, 3);
	//writeArray<float>((float*)solverState.d_p_trans, "p4d_p_trans.txt", nodeN,3);
	//writeArray<float>((float*)solverState.d_rDotzOld, "p4d_rDotzOld.txt", nodeN, 1);

}