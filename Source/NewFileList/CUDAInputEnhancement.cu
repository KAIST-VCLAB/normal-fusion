#include "cuda_SimpleMatrixUtil.h"
#include "RayCastSDFUtil.h"
#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"

#include "InputEnhancementUtil.h"

#include "cudaDebug.h"
#include "CUDAImageUtil.h"
#include "ICPUtil.h"
#include "modeDefine.h"


#define THREADS_PER_BLOCK 512
#define T_PER_BLOCK 32
#define WARP_SIZE 32
#define FLOAT_EPSILON 0.00001f
#define NORM_FLOAT_EPSILON 0.0000001f
#define AVERAGE_SMOOTHNESS_TERM
//#define _DEBUG
#define CENTER_NORMAL
#define PRECONTIDIONER_APPLY
//#define LAPLACIAN_OCTA

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

__inline__ __device__ float warpReduce(float val) {
	int offset = 32 >> 1;
	while (offset > 0) {
		val = val + __shfl_down(val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}

__device__ int2 getPixel2DCoordinate(int idx, InputEnhanceData& state, InputEnhanceParams& params) {
	int2 pixelIdx2D;
	pixelIdx2D.x = idx % params.imageWidth;
	pixelIdx2D.y = idx / params.imageWidth;

	return pixelIdx2D;
}

__device__ float rgbIntensity(const float4 &rgb) {
	return rgb.x * 0.2989f + rgb.y * 0.5870f + rgb.z * 0.1140f;
}

__inline__ __device__ float3 rgbChromacity(const float4 &rgb)
{
	float illumination = rgbIntensity(rgb);
	if (illumination > 0) return make_float3(rgb.x, rgb.y, rgb.z) / illumination;
	return make_float3(0.0f);
}

__inline__ __device__ float huberFunction(const float &a) {
	return 1.f / powf(1.f + 5.f * a, 3.0f);
}

__inline__ __device__ float chromacityWeight(const unsigned int &a, const unsigned int &b, InputEnhanceData& state) {
	float3 chromacity_a = rgbChromacity(state.d_srcImage[a]);
	float3 chromacity_b = rgbChromacity(state.d_srcImage[b]);

	return huberFunction(length(chromacity_a - chromacity_b));
}

__device__ int getPixel1DCoordinate(int2 pixelIdx2D, InputEnhanceData& state, InputEnhanceParams& params) {
	int pixelIdx = -1;

	if (pixelIdx2D.x < 0 || pixelIdx2D.y < 0 || pixelIdx2D.x >= params.imageWidth || pixelIdx2D.y >= params.imageHeight)
		return -1;

	pixelIdx = pixelIdx2D.y * params.imageWidth + pixelIdx2D.x;

	if (!state.d_srcMask[pixelIdx] || state.d_srcDepth[pixelIdx] == MINF)
		return -1;

	return pixelIdx;
}

__device__ float computeLightKernel(float *lightCoeffs, float3 n) {
	float sum = 0;

	sum += lightCoeffs[0];
	sum += lightCoeffs[1] * n.y;
	sum += lightCoeffs[2] * n.z;
	sum += lightCoeffs[3] * n.x;
	sum += lightCoeffs[4] * n.x * n.y;
	sum += lightCoeffs[5] * n.y * n.z;
	sum += lightCoeffs[6] * (-n.x * n.x - n.y * n.y + 2.f * n.z * n.z);
	sum += lightCoeffs[7] * n.z * n.x;
	sum += lightCoeffs[8] * (n.x * n.x - n.y * n.y);

	return fmaxf(0.0f, sum);
}

__device__ float computeDeltaLightKernel(float *lightCoeffs, float *lightDelta, float3 n) {
	float sum = 0;

	sum += lightCoeffs[0] + lightDelta[0];
	sum += (lightCoeffs[1] + lightDelta[1]) * n.y;
	sum += (lightCoeffs[2] + lightDelta[2]) * n.z;
	sum += (lightCoeffs[3] + lightDelta[3]) * n.x;
	sum += (lightCoeffs[4] + lightDelta[4]) * n.x * n.y;
	sum += (lightCoeffs[5] + lightDelta[5]) * n.y * n.z;
	sum += (lightCoeffs[6] + lightDelta[6]) * (-n.x * n.x - n.y * n.y + 2.f * n.z * n.z);
	sum += (lightCoeffs[7] + lightDelta[7]) * n.z * n.x;
	sum += (lightCoeffs[8] + lightDelta[8]) * (n.x * n.x - n.y * n.y);

	return fmaxf(0.0f, sum);
}

__device__ float3 dLightdNormal(float *lightCoeffs, float3 n) {
	float dLdNx =
		lightCoeffs[3] +
		lightCoeffs[4] * n.y +
		(-2.0f * lightCoeffs[6] * n.x) +
		lightCoeffs[7] * n.z +
		2.0f * lightCoeffs[8] * n.x;

	float dLdNy =
		lightCoeffs[1] +
		lightCoeffs[4] * n.x +
		lightCoeffs[5] * n.z +
		(-2.0f * lightCoeffs[6] * n.y) +
		(-2.0f * lightCoeffs[8] * n.y);

	float dLdNz =
		lightCoeffs[2] +
		lightCoeffs[5] * n.y +
		4.0f * lightCoeffs[6] * n.z +
		lightCoeffs[7] * n.x;

	return make_float3(dLdNx, dLdNy, dLdNz);
}

__device__ float3x3 dUNormaldNormal(int point, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	//dtN00N00
	// 1 / (||N||^3) *	[y^2 + z^2, -xy, -xz]
	//					[-xy, x^2 + z^2, -yz]
	//					[-xz, -yz, x^2 + y^2]
	int2 p002D = getPixel2DCoordinate(point, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

#ifdef CENTER_NORMAL
	float d00 = state.d_updateDepth[p00];
	float dn0 = state.d_updateDepth[pn0];
	float d0n = state.d_updateDepth[p0n];
	float dp0 = state.d_updateDepth[pp0];
	float d0p = state.d_updateDepth[p0p];
	float nx = (d0n + d0p) * (dp0 - dn0) / fy;
	float ny = (dn0 + dp0) * (d0p - d0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d0n + d0p) * (dn0 + dp0) / fx / fy));
#else
	float nx = state.d_updateDepth[p0n] * (state.d_updateDepth[p00] - state.d_updateDepth[pn0]) / fy;
	float ny = state.d_updateDepth[pn0] * (state.d_updateDepth[p00] - state.d_updateDepth[p0n]) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - state.d_updateDepth[pn0] * state.d_updateDepth[p0n] / fx / fy);
#endif
	float3 n = make_float3(nx, ny, nz);

	float l = length(n);
	float coeff = 1.0f / (l * l * l);
	float3x3 derivative;
	derivative.setZero();

	derivative(0, 0) = n.y * n.y + n.z * n.z;	derivative(0, 1) = -n.x * n.y;				derivative(0, 2) = -n.x * n.z;
	derivative(1, 0) = -n.x * n.y;				derivative(1, 1) = n.x * n.x + n.z * n.z;	derivative(1, 2) = -n.y * n.z;
	derivative(2, 0) = -n.x * n.z;				derivative(2, 1) = -n.y * n.z;				derivative(2, 2) = n.x * n.x + n.y * n.y;

	return derivative * coeff;
}

__device__ float3 NormalFromDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	if (p0n < 0 || pn0 < 0 || p00 < 0 || p0p < 0 || pp0 < 0) {
		return make_float3(0);
	}

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

#ifdef CENTER_NORMAL
	float d00 = state.d_updateDepth[p00];
	float dn0 = state.d_updateDepth[pn0];
	float d0n = state.d_updateDepth[p0n];
	float dp0 = state.d_updateDepth[pp0];
	float d0p = state.d_updateDepth[p0p];
	float nx = (d0n + d0p) * (dp0 - dn0) / fy;
	float ny = (dn0 + dp0) * (d0p - d0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d0n + d0p) * (dn0 + dp0) / fx / fy));
#else
	float nx = state.d_updateDepth[p0n] * (state.d_updateDepth[p00] - state.d_updateDepth[pn0]) / fy;
	float ny = state.d_updateDepth[pn0] * (state.d_updateDepth[p00] - state.d_updateDepth[p0n]) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - state.d_updateDepth[pn0] * state.d_updateDepth[p0n] / fx / fy);
#endif
	float3 n = make_float3(nx, ny, nz);
	float norm = length(n);

	if (norm < NORM_FLOAT_EPSILON) {
		return make_float3(0);
	}

	return n / norm;
}

__device__ float3 NormalFromDeltaDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = idx;
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	if (p0n < 0 || pn0 < 0 || p00 < 0 || p0p < 0 || pp0 < 0) {
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	float d_p0n = state.d_updateDepth[p0n] + state.d_deltaDepth[p0n];
	float d_pn0 = state.d_updateDepth[pn0] + state.d_deltaDepth[pn0];
	float d_p00 = state.d_updateDepth[p00] + state.d_deltaDepth[p00];

#ifdef CENTER_NORMAL
	float d_pp0 = state.d_updateDepth[pp0] + state.d_deltaDepth[pp0];
	float d_p0p = state.d_updateDepth[p0p] + state.d_deltaDepth[pp0];
	float nx = (d_p0n + d_p0p) * (d_pp0 - d_pn0) / fy;
	float ny = (d_pn0 + d_pp0) * (d_p0p - d_p0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d_p0n + d_p0p) * (d_pn0 + d_pp0) / fx / fy));
#else
	float nx = d_p0n * (d_p00 - d_pn0) / fy;
	float ny = d_pn0 * (d_p00 - d_p0n) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - d_pn0 * d_p0n / fx / fy);
#endif
	float3 n = make_float3(nx, ny, nz);
	float norm = length(n);

	if (norm < NORM_FLOAT_EPSILON) {
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	return n / norm;
}

__device__ float3 CamPosFromDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	if (idx < 0) return make_float3(0.0f);

	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	float posX = (x - cx) / fx;
	float posY = (y - cy) / fy;
	float posZ = 1.0f;

	float3 pos = make_float3(posX, posY, posZ) * state.d_updateDepth[p00];

	return pos;
}

__device__ float3 CamPosFromDeltaDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	float posX = (x - cx) / fx;
	float posY = (y - cy) / fy;
	float posZ = 1.0f;

	float3 pos = make_float3(posX, posY, posZ) * (state.d_updateDepth[p00] + state.d_deltaDepth[p00]);

	return pos;
}

__device__ float NormalLengthFromDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = idx;
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	if (p0n < 0 || pn0 < 0 || p00 < 0 || p0p < 0 || pp0 < 0) {
		return 0;
	}

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

#ifdef CENTER_NORMAL
	float d00 = state.d_updateDepth[p00];
	float dn0 = state.d_updateDepth[pn0];
	float d0n = state.d_updateDepth[p0n];
	float dp0 = state.d_updateDepth[pp0];
	float d0p = state.d_updateDepth[p0p];
	float nx = (d0n + d0p) * (dp0 - dn0) / fy;
	float ny = (dn0 + dp0) * (d0p - d0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d0n + d0p) * (dn0 + dp0) / fx / fy));
#else
	float nx = state.d_updateDepth[p0n] * (state.d_updateDepth[p00] - state.d_updateDepth[pn0]) / fy;
	float ny = state.d_updateDepth[pn0] * (state.d_updateDepth[p00] - state.d_updateDepth[p0n]) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - state.d_updateDepth[pn0] * state.d_updateDepth[p0n] / fx / fy);
#endif

	float3 n = make_float3(nx, ny, nz);
	float norm = length(n);

	if (norm < NORM_FLOAT_EPSILON) {
		return 0;
	}

	return norm;
}

__device__ float NormalLengthFromDeltaDepth(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = idx;
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	if (p0n < 0 || pn0 < 0 || p00 < 0 || p0p < 0 || pp0 < 0) {
		return 0.0f;
	}

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	float d00 = state.d_updateDepth[p00] + state.d_deltaDepth[p00];
	float dn0 = state.d_updateDepth[pn0] + state.d_deltaDepth[pn0];
	float d0n = state.d_updateDepth[p0n] + state.d_deltaDepth[p0n];

#ifdef CENTER_NORMAL
	float dp0 = state.d_updateDepth[pp0] + state.d_deltaDepth[pp0];
	float d0p = state.d_updateDepth[p0p] + state.d_deltaDepth[p0p];
	float nx = (d0n + d0p) * (dp0 - dn0) / fy;
	float ny = (dn0 + dp0) * (d0p - d0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d0n + d0p) * (dn0 + dp0) / fx / fy));
#else
	float nx = d0n * (d00 - dn0) / fy;
	float ny = dn0 * (d00 - d0n) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - dn0 * d0n / fx / fy);
#endif

	float3 n = make_float3(nx, ny, nz);
	float norm = length(n);

	if (norm < NORM_FLOAT_EPSILON) {
		return 0;
	}

	return norm;
}

__inline__ __device__ void evalMinusJTFDeviceShading3(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams, float& resDepth, float& preDepth)
{
	// We apply minus to the final lambda that evalutate minusJTF easily.
	float rDepth = 0.0f;
	float pDepth = 0.0f;

	// Compute -JTF here
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0) return;

	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
	int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
	int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);
	int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);

	float x = p002D.x;
	float y = p002D.y;
	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	// Our target is D(i, j) which is p00.

#ifdef CENTER_NORMAL
	// Eg(i - 1, j) = || B(i - 1, j) - I(i - 1, j) ||
	if (pn0 >= 0 && state.d_optimizeMask[pn0]) {
		float3 albedon0 = make_float3(state.d_updateAlbedo[pn0]);
		float3 cam_nn0 = NormalFromDepth(pn0, state, enhanceParams);
		float3 nn0 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_nn0, 0.0f));

		float3 Bn0 = albedon0 * computeLightKernel(state.d_updateLight, nn0);

		float3 Fn0 = Bn0 - make_float3(state.d_srcImage[pn0]);

		float3 dBn0dNn0_r = albedon0.x * dLightdNormal(state.d_updateLight, nn0);
		float3 dBn0dNn0_g = albedon0.y * dLightdNormal(state.d_updateLight, nn0);
		float3 dBn0dNn0_b = albedon0.z * dLightdNormal(state.d_updateLight, nn0);

		float dNn0dD00_x = (state.d_updateDepth[pnn] + state.d_updateDepth[pnp]) / fy;
		float dNn0dD00_y = (state.d_updateDepth[pnp] - state.d_updateDepth[pnn]) / fx;
		float dNn0dD00_z = ((dNn0dD00_x * (cx - (x - 1.0f)) / fx) + (dNn0dD00_y * (cy - y) / fy) - ((state.d_updateDepth[pnn] + state.d_updateDepth[pnp]) / fx / fy));

		float3 dNn0dD00 = dUNormaldNormal(pn0, state, enhanceParams) * make_float3(dNn0dD00_x, dNn0dD00_y, dNn0dD00_z);

		// D(i, j) is in B(i - 1, j)
		// dB(i - 1, j) / dD(i, j)
		float3 dBn0dD00 = make_float3(dot(dBn0dNn0_r, dNn0dD00), dot(dBn0dNn0_g, dNn0dD00), dot(dBn0dNn0_b, dNn0dD00));
		float dBn0dD00_dot_Fn0 = dot(dBn0dD00, Fn0);
		rDepth += dBn0dD00_dot_Fn0;
		pDepth += enhanceParams.weightDataShading * dot(dBn0dD00, dBn0dD00);
	}

	// Eg(i, j - 1) = || B(i, j - 1) - I(i, j - 1) ||
	if (p0n >= 0 && state.d_optimizeMask[p0n]) {
		float3 albedo0n = make_float3(state.d_updateAlbedo[p0n]);
		float3 cam_n0n = NormalFromDepth(p0n, state, enhanceParams);
		float3 n0n = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n0n, 0.0f));

		float3 B0n = albedo0n * computeLightKernel(state.d_updateLight, n0n);

		float3 F0n = B0n - make_float3(state.d_srcImage[p0n]);

		float3 dB0ndN0n_r = albedo0n.x * dLightdNormal(state.d_updateLight, n0n);
		float3 dB0ndN0n_g = albedo0n.y * dLightdNormal(state.d_updateLight, n0n);
		float3 dB0ndN0n_b = albedo0n.z * dLightdNormal(state.d_updateLight, n0n);

		float dN0ndD00_x = (state.d_updateDepth[ppn] - state.d_updateDepth[pnn]) / fy;
		float dN0ndD00_y = (state.d_updateDepth[pnn] + state.d_updateDepth[ppn]) / fx;
		float dN0ndD00_z = ((dN0ndD00_x * (cx - x) / fx) + (dN0ndD00_y * (cy - (y - 1.0f)) / fy) - ((state.d_updateDepth[pnn] + state.d_updateDepth[ppn]) / fx / fy));

		float3 dN0ndD00 = dUNormaldNormal(p0n, state, enhanceParams) * make_float3(dN0ndD00_x, dN0ndD00_y, dN0ndD00_z);

		// D(i, j) is in B(i, j - 1)
		// dB(i, j - 1) / dD(i, j)
		float3 dB0ndD00 = make_float3(dot(dB0ndN0n_r, dN0ndD00), dot(dB0ndN0n_g, dN0ndD00), dot(dB0ndN0n_b, dN0ndD00));
		float dB0ndD00_dot_F0n = dot(dB0ndD00, F0n);
		rDepth += dB0ndD00_dot_F0n;
		pDepth += enhanceParams.weightDataShading * dot(dB0ndD00, dB0ndD00);
	}

	// Eg(i + 1, j) = || B(i + 1, j) - I(i + 1, j) ||
	if (pp0 >= 0 && state.d_optimizeMask[pp0]) {
		float3 albedop0 = make_float3(state.d_updateAlbedo[pp0]);
		float3 cam_np0 = NormalFromDepth(pp0, state, enhanceParams);
		float3 np0 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_np0, 0.0f));

		float3 Bp0 = albedop0 * computeLightKernel(state.d_updateLight, np0);

		float3 Fp0 = Bp0 - make_float3(state.d_srcImage[pp0]);

		float3 dBp0dNp0_r = albedop0.x * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_g = albedop0.y * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_b = albedop0.z * dLightdNormal(state.d_updateLight, np0);

		float dNp0dD00_x = -(state.d_updateDepth[ppn] + state.d_updateDepth[ppp]) / fy;
		float dNp0dD00_y = (state.d_updateDepth[ppp] - state.d_updateDepth[ppn]) / fx;
		float dNp0dD00_z = ((dNp0dD00_x * (cx - (x + 1.0f)) / fx) + (dNp0dD00_y * (cy - y) / fy) - ((state.d_updateDepth[ppn] + state.d_updateDepth[ppp]) / fx / fy));

		float3 dNp0dD00 = dUNormaldNormal(pp0, state, enhanceParams) * make_float3(dNp0dD00_x, dNp0dD00_y, dNp0dD00_z);

		// D(i, j) is in B(i + 1, j)
		// dB(i + 1, j) / dD(i, j)
		float3 dBp0dD00 = make_float3(dot(dBp0dNp0_r, dNp0dD00), dot(dBp0dNp0_g, dNp0dD00), dot(dBp0dNp0_b, dNp0dD00));
		float dBp0dD00_dot_Fp0 = dot(dBp0dD00, Fp0);
		rDepth += dBp0dD00_dot_Fp0;
		pDepth += enhanceParams.weightDataShading * dot(dBp0dD00, dBp0dD00);
	}

	// Eg(i, j + 1) = || B(i, j + 1) - I(i, j + 1) ||
	if (p0p >= 0 && state.d_optimizeMask[p0p]) {
		float3 albedo0p = make_float3(state.d_updateAlbedo[p0p]);
		float3 cam_n0p = NormalFromDepth(p0p, state, enhanceParams);
		float3 n0p = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n0p, 0.0f));

		float3 B0p = albedo0p * computeLightKernel(state.d_updateLight, n0p);

		float3 F0p = B0p - make_float3(state.d_srcImage[p0p]);

		float3 dB0pdN0p_r = albedo0p.x * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_g = albedo0p.y * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_b = albedo0p.z * dLightdNormal(state.d_updateLight, n0p);

		float dN0pdD00_x = (state.d_updateDepth[ppp] - state.d_updateDepth[pnp]) / fy;
		float dN0pdD00_y = -(state.d_updateDepth[pnp] + state.d_updateDepth[ppp]) / fx;
		float dN0pdD00_z = ((dN0pdD00_x * (cx - x) / fx) + (dN0pdD00_y * (cy - (y + 1.0f)) / fy) - ((state.d_updateDepth[pnp] + state.d_updateDepth[ppp]) / fx / fy));

		float3 dN0pdD00 = dUNormaldNormal(p0p, state, enhanceParams) * make_float3(dN0pdD00_x, dN0pdD00_y, dN0pdD00_z);

		// D(i, j) is in B(i, j + 1)
		// dB(i, j + 1) / dD(i, j)
		float3 dB0pdD00 = make_float3(dot(dB0pdN0p_r, dN0pdD00), dot(dB0pdN0p_g, dN0pdD00), dot(dB0pdN0p_b, dN0pdD00));
		float dB0pdD00_dot_F0p = dot(dB0pdD00, F0p);
		rDepth += dB0pdD00_dot_F0p;
		pDepth += enhanceParams.weightDataShading * dot(dB0pdD00, dB0pdD00);
	}
#else
	// Eg(i, j) = || B(i, j) - I(i, j) ||
	if (state.d_iterMask[p00]) {
		float3 albedo00 = state.d_updateAlbedo[p00];
		float3 n00 = NormalFromDepth(p00, state, enhanceParams);
		float3 B00 = albedo00 * computeLightKernel(state.d_updateLight, n00);

		float3 F00 = B00 - state.d_srcImage[p00];

		float3 dB00dN00_r = albedo00.x * dLightdNormal(state.d_updateLight, n00);
		float3 dB00dN00_g = albedo00.y * dLightdNormal(state.d_updateLight, n00);
		float3 dB00dN00_b = albedo00.z * dLightdNormal(state.d_updateLight, n00);

		float dN00dD00_x = state.d_updateDepth[p0n] / fy;
		float dN00dD00_y = state.d_updateDepth[pn0] / fx;
		float dN00dD00_z = (dN00dD00_x * (cx - x) / fx + dN00dD00_y * (cy - y) / fy);
		float3 dN00dD00 = dUNormaldNormal(p00, state, enhanceParams) * make_float3(dN00dD00_x, dN00dD00_y, dN00dD00_z);

		float3 dB00dD00 = make_float3(dot(dB00dN00_r, dN00dD00), dot(dB00dN00_g, dN00dD00), dot(dB00dN00_b, dN00dD00));
		// D(i, j) is in ||B(i, j) - B(i + 1, j)||
		// dB(i, j) / dD(i, j)
		float dB00dD00_dot_F00 = dot(dB00dD00, F00);
		rDepth += dB00dD00_dot_F00;
		pDepth += enhanceParams.weightDataShading * dot(dB00dD00, dB00dD00);
	}

	// Eg(i, j + 1) = || B(i, j + 1) - I(i, j + 1) ||
	if (p0p >= 0 && state.d_iterMask[p0p]) {
		float3 albedo0p = state.d_updateAlbedo[p0p];
		float3 n0p = NormalFromDepth(p0p, state, enhanceParams);
		float3 B0p = albedo0p * computeLightKernel(state.d_updateLight, n0p);

		float3 F0p = B0p - state.d_srcImage[p0p];

		float3 dB0pdN0p_r = albedo0p.x * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_g = albedo0p.y * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_b = albedo0p.z * dLightdNormal(state.d_updateLight, n0p);

		float dN0pdD00_x = (state.d_updateDepth[p0p] - state.d_updateDepth[pnp]) / fy;
		float dN0pdD00_y = -state.d_updateDepth[pnp] / fx;
		float dN0pdD00_z = (dN0pdD00_x * (cx - x) / fx + dN0pdD00_y * (cy - (y + 1.0f)) / fy - state.d_updateDepth[pnp] / fx / fy);
		float3 dN0pdD00 = dUNormaldNormal(p0p, state, enhanceParams) * make_float3(dN0pdD00_x, dN0pdD00_y, dN0pdD00_z);

		// D(i, j) is in B(i, j + 1)
		// dB(i, j + 1) / dD(i, j)
		float3 dB0pdD00 = make_float3(dot(dB0pdN0p_r, dN0pdD00), dot(dB0pdN0p_g, dN0pdD00), dot(dB0pdN0p_b, dN0pdD00));
		float dB0pdD00_dot_F0p = dot(dB0pdD00, F0p);
		rDepth += dB0pdD00_dot_F0p;
		pDepth += enhanceParams.weightDataShading * dot(dB0pdD00, dB0pdD00);
	}

	// Eg(i + 1, j) = || B(i + 1, j) - I(i + 1, j) ||
	if (pp0 >= 0 && state.d_iterMask[pp0]) {
		float3 albedop0 = state.d_updateAlbedo[pp0];
		float3 np0 = NormalFromDepth(pp0, state, enhanceParams);
		float3 Bp0 = albedop0 * computeLightKernel(state.d_updateLight, np0);

		float3 Fp0 = Bp0 - state.d_srcImage[pp0];

		float3 dBp0dNp0_r = albedop0.x * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_g = albedop0.y * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_b = albedop0.z * dLightdNormal(state.d_updateLight, np0);

		float dNp0dD00_x = (-1.0f * state.d_updateDepth[ppn] / fy);
		float dNp0dD00_y = (state.d_updateDepth[pp0] - state.d_updateDepth[ppn]) / fx;
		float dNp0dD00_z = (dNp0dD00_x * (cx - (x + 1.0f)) / fx + dNp0dD00_y * (cy - y) / fy - state.d_updateDepth[ppn] / fx / fy);
		float3 dNp0dD00 = dUNormaldNormal(pp0, state, enhanceParams) * make_float3(dNp0dD00_x, dNp0dD00_y, dNp0dD00_z);

		// D(i, j) is in B(i + 1, j)
		// dB(i + 1, j) / dD(i, j)
		float3 dBp0dD00 = make_float3(dot(dBp0dNp0_r, dNp0dD00), dot(dBp0dNp0_g, dNp0dD00), dot(dBp0dNp0_b, dNp0dD00));
		float dBp0dD00_dot_Fp0 = dot(dBp0dD00, Fp0);
		rDepth += dBp0dD00_dot_Fp0;
		pDepth += enhanceParams.weightDataShading * dot(dBp0dD00, dBp0dD00);
	}
#endif
	resDepth += -enhanceParams.weightDataShading * rDepth;
	preDepth += pDepth;
}

__inline__ __device__ void evalMinusJTFDeviceReg(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, float& resDepth, float& preDepth)
{
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 >= 0 && state.d_optimizeMask[idx]) {
		resDepth += -enhanceParams.weightDepthTemp * (state.d_updateDepth[p00] - state.d_srcDepth[p00]);
		preDepth += enhanceParams.weightDepthTemp;
	}
}

__inline__ __device__ void evalMinusJTFDeviceSmooth(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, float& resDepth, float& preDepth)
{
	float rDepth = 0.0f;
	float pDepth = 0.0f;

	// Compute -JTF here
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0) return;

	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
	int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
	int pm0 = getPixel1DCoordinate(p002D + make_int2(-2, 0), state, enhanceParams);
	int p0m = getPixel1DCoordinate(p002D + make_int2(0, -2), state, enhanceParams);
	int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);
	int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);
	int pq0 = getPixel1DCoordinate(p002D + make_int2(2, 0), state, enhanceParams);
	int p0q = getPixel1DCoordinate(p002D + make_int2(0, 2), state, enhanceParams);

	// Smoothness term || D(i, j) - D_k(i, j) ||
	float Fn0_smoothness = 0.0f;
	float F0n_smoothness = 0.0f;
	float Fp0_smoothness = 0.0f;
	float F0p_smoothness = 0.0f;

	if (state.d_optimizeMask[p00]) {
#ifdef LAPLACIAN_OCTA
		rDepth = state.d_updateDepth[p00] -
			0.125f * (state.d_updateDepth[pn0] + state.d_updateDepth[p0n] + state.d_updateDepth[pp0] + state.d_updateDepth[p0p]
				+ state.d_updateDepth[ppn] + state.d_updateDepth[pnp] + state.d_updateDepth[ppp] + state.d_updateDepth[pnn]);
		pDepth += enhanceParams.weightDepthSmooth;
#else
		rDepth = state.d_updateDepth[p00] -
			0.25f * (state.d_updateDepth[pn0] + state.d_updateDepth[p0n] + state.d_updateDepth[pp0] + state.d_updateDepth[p0p]);
		pDepth += enhanceParams.weightDepthSmooth;
#endif
	}


	float weight = 0.0f;
	if (pn0 >= 0 && state.d_optimizeMask[pn0]) {
#ifdef LAPLACIAN_OCTA
			int pmp = getPixel1DCoordinate(p002D + make_int2(-2, 1), state, enhanceParams);
			int pmn = getPixel1DCoordinate(p002D + make_int2(-2, -1), state, enhanceParams);
			weight += 1.0f;
			Fn0_smoothness = state.d_updateDepth[pn0] -
				0.125f * (state.d_updateDepth[pm0] + state.d_updateDepth[pnn] + state.d_updateDepth[p00] + state.d_updateDepth[pnp]
					+ state.d_updateDepth[p0n] + state.d_updateDepth[p0p] + state.d_updateDepth[pmp] + state.d_updateDepth[pmn]);
#else
			weight += 1.0f;
			Fn0_smoothness = state.d_updateDepth[pn0] -
				0.25f * (state.d_updateDepth[pm0] + state.d_updateDepth[pnn] + state.d_updateDepth[p00] + state.d_updateDepth[pnp]);
#endif
	}
	if (p0n >= 0 && state.d_optimizeMask[p0n]) {
#ifdef LAPLACIAN_OCTA
			int ppm = getPixel1DCoordinate(p002D + make_int2(1, -2), state, enhanceParams);
			int pnm = getPixel1DCoordinate(p002D + make_int2(-1, -2), state, enhanceParams);
			weight += 1.0f;
			F0n_smoothness = state.d_updateDepth[p0n] -
				0.125f * (state.d_updateDepth[pnn] + state.d_updateDepth[p0m] + state.d_updateDepth[ppn] + state.d_updateDepth[p00]
					+ state.d_updateDepth[ppm] + state.d_updateDepth[pp0] + state.d_updateDepth[pn0] + state.d_updateDepth[pnm]);
#else
			weight += 1.0f;
			F0n_smoothness = state.d_updateDepth[p0n] -
				0.25f * (state.d_updateDepth[pnn] + state.d_updateDepth[p0m] + state.d_updateDepth[ppn] + state.d_updateDepth[p00]);
#endif
	}
	if (pp0 >= 0 &&state.d_optimizeMask[pp0]) {
#ifdef LAPLACIAN_OCTA
			int pqn = getPixel1DCoordinate(p002D + make_int2(2, -1), state, enhanceParams);
			int pqp = getPixel1DCoordinate(p002D + make_int2(2, 1), state, enhanceParams);
			weight += 1.0f;
			Fp0_smoothness = state.d_updateDepth[pp0] -
				0.125f * (state.d_updateDepth[p00] + state.d_updateDepth[ppn] + state.d_updateDepth[pq0] + state.d_updateDepth[ppp]
					+ state.d_updateDepth[pqn] + state.d_updateDepth[pqp] + state.d_updateDepth[p0p] + state.d_updateDepth[p0n]);
#else
			weight += 1.0f;
			Fp0_smoothness = state.d_updateDepth[pp0] -
				0.25f * (state.d_updateDepth[p00] + state.d_updateDepth[ppn] + state.d_updateDepth[pq0] + state.d_updateDepth[ppp]);
#endif
	}
	if (p0p >= 0 && state.d_optimizeMask[p0p]) {
#ifdef LAPLACIAN_OCTA
			int ppq = getPixel1DCoordinate(p002D + make_int2(1, 2), state, enhanceParams);
			int pnq = getPixel1DCoordinate(p002D + make_int2(-1, 2), state, enhanceParams);
			weight += 1.0f;
			F0p_smoothness = state.d_updateDepth[p0p] -
				0.125f * (state.d_updateDepth[pnp] + state.d_updateDepth[p00] + state.d_updateDepth[ppp] + state.d_updateDepth[p0q]
					+ state.d_updateDepth[pp0] + state.d_updateDepth[ppq] + state.d_updateDepth[pnq] + state.d_updateDepth[pn0]);
#else
			weight += 1.0f;
			F0p_smoothness = state.d_updateDepth[p0p] -
				0.25f * (state.d_updateDepth[pnp] + state.d_updateDepth[p00] + state.d_updateDepth[ppp] + state.d_updateDepth[p0q]);
#endif
	}


#ifdef LAPLACIAN_OCTA
	float Fpn_smoothness = 0.0f;
	float Fpp_smoothness = 0.0f;
	float Fnp_smoothness = 0.0f;
	float Fnn_smoothness = 0.0f;
	if (ppn >= 0 && state.d_optimizeMask[ppn]) {
		int ppm = getPixel1DCoordinate(p002D + make_int2(1, -2), state, enhanceParams);
		int pqn = getPixel1DCoordinate(p002D + make_int2(2, -1), state, enhanceParams);
		int pqm = getPixel1DCoordinate(p002D + make_int2(2, -2), state, enhanceParams);
		weight += 1.0f;
		Fpn_smoothness = state.d_updateDepth[ppn] -
			0.125f * (state.d_updateDepth[p0n] + state.d_updateDepth[ppm] + state.d_updateDepth[pqn] + state.d_updateDepth[pp0]
				+ state.d_updateDepth[pqm] + state.d_updateDepth[pq0] + state.d_updateDepth[p00] + state.d_updateDepth[p0m]);
	}
	if (ppp >= 0 && state.d_optimizeMask[ppp]) {
		int pqp = getPixel1DCoordinate(p002D + make_int2(2, 1), state, enhanceParams);
		int ppq = getPixel1DCoordinate(p002D + make_int2(1, 2), state, enhanceParams);
		int pqq = getPixel1DCoordinate(p002D + make_int2(2, 2), state, enhanceParams);
		weight += 1.0f;
		Fpp_smoothness = state.d_updateDepth[ppp] -
			0.125f * (state.d_updateDepth[p0p] + state.d_updateDepth[pp0] + state.d_updateDepth[pqp] + state.d_updateDepth[ppq]
				+ state.d_updateDepth[pq0] + state.d_updateDepth[pqq] + state.d_updateDepth[p0q] + state.d_updateDepth[p00]);
	}
	if (pnp >= 0 && state.d_optimizeMask[pnp]) {
		int pmp = getPixel1DCoordinate(p002D + make_int2(-2, 1), state, enhanceParams);
		int pnq = getPixel1DCoordinate(p002D + make_int2(-1, 2), state, enhanceParams);
		int pmq = getPixel1DCoordinate(p002D + make_int2(-2, 2), state, enhanceParams);
		weight += 1.0f;
		Fnp_smoothness = state.d_updateDepth[pnp] -
			0.125f * (state.d_updateDepth[pmp] + state.d_updateDepth[pn0] + state.d_updateDepth[p0p] + state.d_updateDepth[pnq]
				+ state.d_updateDepth[p00] + state.d_updateDepth[p0q] + state.d_updateDepth[pmq] + state.d_updateDepth[pm0]);
	}
	if (pnn >= 0 && state.d_optimizeMask[pnn]) {
		int pnm = getPixel1DCoordinate(p002D + make_int2(-1, -2), state, enhanceParams);
		int pmn = getPixel1DCoordinate(p002D + make_int2(-2, -1), state, enhanceParams);
		int pmm = getPixel1DCoordinate(p002D + make_int2(-2, -2), state, enhanceParams);
		weight += 1.0f;
		Fnn_smoothness = state.d_updateDepth[pnn] -
			0.125f * (state.d_updateDepth[pmn] + state.d_updateDepth[pnm] + state.d_updateDepth[p0n] + state.d_updateDepth[pn0]
				+ state.d_updateDepth[p0m] + state.d_updateDepth[p00] + state.d_updateDepth[pm0] + state.d_updateDepth[pmm]);
	}
	rDepth += -0.125f * (Fn0_smoothness + F0n_smoothness + Fp0_smoothness + F0p_smoothness + Fpn_smoothness + Fpp_smoothness + Fnp_smoothness + Fnn_smoothness);
	pDepth += enhanceParams.weightDepthSmooth * (0.125f * 0.125f * weight);
#else 
	rDepth += -0.25f * (Fn0_smoothness + F0n_smoothness + Fp0_smoothness + F0p_smoothness);
	pDepth += enhanceParams.weightDepthSmooth * (0.25f * 0.25f * weight);
#endif

	resDepth += -enhanceParams.weightDepthSmooth * rDepth;
	preDepth += pDepth;
}

__inline__ __device__ void evalMinusJTFDeviceShadingAlbedo(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams, float3& resAlbedo, float3& preAlbedo)
{
	// We apply minus to the final lambda that evalutate minusJTF easily.
	float3 rAlbedo = make_float3(0);
	float3 pAlbedo = make_float3(0);

	// Compute -JTF here
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) return;

	// Our target is A(i, j) which is p00.

	// Eg(i, j) = || B(i, j) - I(i, j) ||
	float3 albedo00 = make_float3(state.d_updateAlbedo[p00]);
	float3 cam_n00 = NormalFromDepth(p00, state, enhanceParams);
	float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));

	float3 B00 = albedo00 * computeLightKernel(state.d_updateLight, n00);

	float3 F00 = B00 - make_float3(state.d_srcImage[p00]);

	float3 dB00dA00 = make_float3(computeLightKernel(state.d_updateLight, n00));

	rAlbedo += F00 * dB00dA00;
	pAlbedo += enhanceParams.weightDataShading * (dB00dA00 * dB00dA00);

	resAlbedo += -enhanceParams.weightDataShading * rAlbedo;
	preAlbedo += pAlbedo;
}

__inline__ __device__ void evalMinusJTFDeviceAlbedoTemp(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, float3& resAlbedo, float3& preAlbedo)
{
	// Compute -JTF here
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx] || !state.d_albedoTempMask[idx]) return;

	// Our target is A(i, j) which is p00.

	// Eg(i, j) = || B(i, j) - I(i, j) ||
	float3 albedo00 = make_float3(state.d_updateAlbedo[p00]);
	float3 srcAlbedo00 = make_float3(state.d_prevAlbedo[p00]);

	float3 F00 = albedo00 - srcAlbedo00;

	resAlbedo += -enhanceParams.weightAlbedoTemp * F00;
	preAlbedo += make_float3(enhanceParams.weightAlbedoTemp);
}

__inline__ __device__ void evalMinusJTFDeviceAlbedoSpatial(unsigned int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, float3& resAlbedo, float3& preAlbedo)
{
	// We apply minus to the final lambda that evalutate minusJTF easily.
	float3 rAlbedo = make_float3(0);
	float3 pAlbedo = make_float3(0);

	// Compute -JTF here
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0) return;

	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);


	// Our target is A(i, j) which is p00.

	// Eg(i, j) = || A(i, j) - An(i, j) ||
	float3 F00_n0 = make_float3(0);
	float3 F00_0n = make_float3(0);
	float3 F00_p0 = make_float3(0);
	float3 F00_0p = make_float3(0);

	if (state.d_optimizeMask[p00]) {
		float3 albedo00 = make_float3(state.d_updateAlbedo[p00]);
		float3 albedon0 = make_float3(state.d_updateAlbedo[pn0]);
		float3 albedo0n = make_float3(state.d_updateAlbedo[p0n]);
		float3 albedop0 = make_float3(state.d_updateAlbedo[pp0]);
		float3 albedo0p = make_float3(state.d_updateAlbedo[p0p]);

		float weightn0 = chromacityWeight(p00, pn0, state);
		float weight0n = chromacityWeight(p00, p0n, state);
		float weightp0 = chromacityWeight(p00, pp0, state);
		float weight0p = chromacityWeight(p00, p0p, state);

		F00_n0 = weightn0 * (albedo00 - albedon0);
		F00_0n = weight0n * (albedo00 - albedo0n);
		F00_p0 = weightp0 * (albedo00 - albedop0);
		F00_0p = weight0p * (albedo00 - albedo0p);

		pAlbedo += enhanceParams.weightAlbedoSmooth * make_float3(weightn0 + weight0n + weightp0 + weight0p);
	}

	float3 Fn0_00 = make_float3(0);
	float3 F0n_00 = make_float3(0);
	float3 Fp0_00 = make_float3(0);
	float3 F0p_00 = make_float3(0);

	float validWeight = 0.0f;
	if (pn0 >= 0 && state.d_optimizeMask[pn0]) { Fn0_00 = chromacityWeight(pn0, p00, state) * make_float3(state.d_updateAlbedo[pn0] - state.d_updateAlbedo[p00]); validWeight += chromacityWeight(pn0, p00, state); }
	if (p0n >= 0 && state.d_optimizeMask[p0n]) { F0n_00 = chromacityWeight(p0n, p00, state) * make_float3(state.d_updateAlbedo[p0n] - state.d_updateAlbedo[p00]); validWeight += chromacityWeight(p0n, p00, state); }
	if (pp0 >= 0 && state.d_optimizeMask[pp0]) { Fp0_00 = chromacityWeight(pp0, p00, state) * make_float3(state.d_updateAlbedo[pp0] - state.d_updateAlbedo[p00]); validWeight += chromacityWeight(pp0, p00, state); }
	if (p0p >= 0 && state.d_optimizeMask[p0p]) { F0p_00 = chromacityWeight(p0p, p00, state) * make_float3(state.d_updateAlbedo[p0p] - state.d_updateAlbedo[p00]); validWeight += chromacityWeight(p0p, p00, state); }

	rAlbedo += (F00_n0 + F00_0n + F00_p0 + F00_0p - Fn0_00 - F0n_00 - Fp0_00 - F0p_00);
	pAlbedo += enhanceParams.weightAlbedoSmooth * make_float3(validWeight);

	resAlbedo += -enhanceParams.weightAlbedoSmooth * rAlbedo;
	preAlbedo += pAlbedo;
}


__device__ float3 applyJpShading3DeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) {
		return make_float3(0);
	}

	float p_p00 = state.d_pDepth[p00];
	float p_p0n = state.d_pDepth[p0n];
	float p_pn0 = state.d_pDepth[pn0];
	float p_p0p = state.d_pDepth[p0p];
	float p_pp0 = state.d_pDepth[pp0];

	float x = p002D.x;
	float y = p002D.y;
	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

	// B(i, j)
	float3 albedo00 = make_float3(state.d_updateAlbedo[p00]);
	float3 cam_n00 = NormalFromDepth(p00, state, enhanceParams);
	float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));
	float3 dB00dN00_r = albedo00.x * dLightdNormal(state.d_updateLight, n00);
	float3 dB00dN00_g = albedo00.y * dLightdNormal(state.d_updateLight, n00);
	float3 dB00dN00_b = albedo00.z * dLightdNormal(state.d_updateLight, n00);
	float3x3 dUNormal00dNormal00 = dUNormaldNormal(p00, state, enhanceParams);
#ifdef CENTER_NORMAL
	// dB(i, j) / dD(i - 1, j)
	float dN00dDn0_x = -(state.d_updateDepth[p0n] + state.d_updateDepth[p0p]) / fy;
	float dN00dDn0_y = (state.d_updateDepth[p0p] - state.d_updateDepth[p0n]) / fx;
	float dN00dDn0_z = ((dN00dDn0_x * (cx - x) / fx) + (dN00dDn0_y * (cy - y) / fy) - ((state.d_updateDepth[p0n] + state.d_updateDepth[p0p]) / fx / fy));
	float3 dN00dDn0 = dUNormal00dNormal00 * make_float3(dN00dDn0_x, dN00dDn0_y, dN00dDn0_z);
	float3 dB00dDn0 = make_float3(dot(dB00dN00_r, dN00dDn0), dot(dB00dN00_g, dN00dDn0), dot(dB00dN00_b, dN00dDn0));
	float3 dB00dDn0_p_pn0 = dB00dDn0 * p_pn0;

	// dB(i, j) / dD(i, j - 1)
	float dN00dD0n_x = (state.d_updateDepth[pp0] - state.d_updateDepth[pn0]) / fy;
	float dN00dD0n_y = -(state.d_updateDepth[pn0] + state.d_updateDepth[pp0]) / fx;
	float dN00dD0n_z = ((dN00dD0n_x * (cx - x) / fx) + (dN00dD0n_y * (cy - y) / fy) - ((state.d_updateDepth[pn0] + state.d_updateDepth[pp0]) / fx / fy));
	float3 dN00dD0n = dUNormal00dNormal00 * make_float3(dN00dD0n_x, dN00dD0n_y, dN00dD0n_z);
	float3 dB00dD0n = make_float3(dot(dB00dN00_r, dN00dD0n), dot(dB00dN00_g, dN00dD0n), dot(dB00dN00_b, dN00dD0n));
	float3 dB00dD0n_p_p0n = dB00dD0n * p_p0n;

	// dB(i, j) / dD(i + 1, j)
	float dN00dDp0_x = (state.d_updateDepth[p0n] + state.d_updateDepth[p0p]) / fy;
	float dN00dDp0_y = (state.d_updateDepth[p0p] - state.d_updateDepth[p0n]) / fx;
	float dN00dDp0_z = ((dN00dDp0_x * (cx - x) / fx) + (dN00dDp0_y * (cy - y) / fy) - ((state.d_updateDepth[p0n] + state.d_updateDepth[p0p]) / fx / fy));
	float3 dN00dDp0 = dUNormal00dNormal00 * make_float3(dN00dDp0_x, dN00dDp0_y, dN00dDp0_z);
	float3 dB00dDp0 = make_float3(dot(dB00dN00_r, dN00dDp0), dot(dB00dN00_g, dN00dDp0), dot(dB00dN00_b, dN00dDp0));
	float3 dB00dDp0_p_pp0 = dB00dDp0 * p_pp0;

	// dB(i, j) / dD(i, j + 1)
	float dN00dD0p_x = (state.d_updateDepth[pp0] - state.d_updateDepth[pn0]) / fy;
	float dN00dD0p_y = (state.d_updateDepth[pn0] + state.d_updateDepth[pp0]) / fx;
	float dN00dD0p_z = ((dN00dD0p_x * (cx - x) / fx) + (dN00dD0p_y * (cy - y) / fy) - ((state.d_updateDepth[pn0] + state.d_updateDepth[pp0]) / fx / fy));
	float3 dN00dD0p = dUNormal00dNormal00 * make_float3(dN00dD0p_x, dN00dD0p_y, dN00dD0p_z);
	float3 dB00dD0p = make_float3(dot(dB00dN00_r, dN00dD0p), dot(dB00dN00_g, dN00dD0p), dot(dB00dN00_b, dN00dD0p));
	float3 dB00dD0p_p_p0p = dB00dD0p * p_p0p;

	return dB00dDn0_p_pn0 + dB00dD0n_p_p0n + dB00dDp0_p_pp0 + dB00dD0p_p_p0p;
#else
	// dB(i, j) / dD(i - 1, j)
	float dN00dDn0_x = -1.0f * state.d_updateDepth[p0n] / fy;
	float dN00dDn0_y = (state.d_updateDepth[p00] - state.d_updateDepth[p0n]) / fx;
	float dN00dDn0_z = (dN00dDn0_x * (cx - x) / fx + dN00dDn0_y * (cy - y) / fy - state.d_updateDepth[p0n] / fx / fy);
	float3 dN00dDn0 = dUNormal00dNormal00 * make_float3(dN00dDn0_x, dN00dDn0_y, dN00dDn0_z);
	float3 dB00dDn0 = make_float3(dot(dB00dN00_r, dN00dDn0), dot(dB00dN00_g, dN00dDn0), dot(dB00dN00_b, dN00dDn0));
	float3 dB00dDn0_p_pn0 = dB00dDn0 * p_pn0;

	// dB(i, j) / dD(i, j)
	float dN00dD00_x = state.d_updateDepth[p0n] / fy;
	float dN00dD00_y = state.d_updateDepth[pn0] / fx;
	float dN00dD00_z = (dN00dD00_x * (cx - x) / fx + dN00dD00_y * (cy - y) / fy);
	float3 dN00dD00 = dUNormal00dNormal00 * make_float3(dN00dD00_x, dN00dD00_y, dN00dD00_z);
	float3 dB00dD00 = make_float3(dot(dB00dN00_r, dN00dD00), dot(dB00dN00_g, dN00dD00), dot(dB00dN00_b, dN00dD00));
	float3 dB00dD00_p_p00 = dB00dD00 * p_p00;

	// dB(i, j) / dD(i, j - 1)
	float dN00dD0n_x = (state.d_updateDepth[p00] - state.d_updateDepth[pn0]) / fy;
	float dN00dD0n_y = -1.0f * state.d_updateDepth[pn0] / fx;
	float dN00dD0n_z = (dN00dD0n_x * (cx - x) / fx + dN00dD0n_y * (cy - y) / fy - state.d_updateDepth[pn0] / fx / fy);
	float3 dN00dD0n = dUNormal00dNormal00 * make_float3(dN00dD0n_x, dN00dD0n_y, dN00dD0n_z);
	float3 dB00dD0n = make_float3(dot(dB00dN00_r, dN00dD0n), dot(dB00dN00_g, dN00dD0n), dot(dB00dN00_b, dN00dD0n));
	float3 dB00dD0n_p_p0n = dB00dD0n * p_p0n;

	return dB00dDn0_p_pn0 + dB00dD00_p_p00 + dB00dD0n_p_p0n;
#endif
}

// JpReg kernel call
__device__ float applyJpRegDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) return 0.0f;
	return state.d_pDepth[p00];
}

// JpSmooth kernel call
__device__ void applyJpSmoothDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	// Normal orientation, we have to minus x and z.
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
#ifdef LAPLACIAN_OCTA
	int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
	int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);
	int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
	int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);
#endif

	if (p00 < 0 || !state.d_optimizeMask[idx]) {
		state.d_Jp_smooth[idx] = 0.0f;
		return;
	}

	float p_p00 = state.d_pDepth[p00];
	float p_pn0 = state.d_pDepth[pn0];
	float p_p0n = state.d_pDepth[p0n];
	float p_pp0 = state.d_pDepth[pp0];
	float p_p0p = state.d_pDepth[p0p];
#ifdef LAPLACIAN_OCTA
	float p_ppn = state.d_pDepth[ppn];
	float p_ppp = state.d_pDepth[ppp];
	float p_pnp = state.d_pDepth[pnp];
	float p_pnn = state.d_pDepth[pnn];

	state.d_Jp_smooth[idx] = p_p00 - 0.125f * (p_pn0 + p_p0n + p_pp0 + p_p0p + p_ppn + p_ppp + p_pnp + p_pnn);
#else
	state.d_Jp_smooth[p00] = p_p00 - 0.25f * (p_pn0 + p_p0n + p_pp0 + p_p0p);
#endif
}

__device__ float3 applyJpShadingAlbedoDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) {
		return make_float3(0.0f);
	}

	float3 p_p00 = state.d_pAlbedo[p00];

	float3 cam_n00 = NormalFromDepth(p00, state, enhanceParams);
	float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));

	float3 dB00dA00 = make_float3(computeLightKernel(state.d_updateLight, n00));
	float3 dB00dA00_dot_p_p00 = dB00dA00 * p_p00;

	return dB00dA00_dot_p_p00;
}

__device__ float3 applyJpAlbedoTempDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx] || !state.d_albedoTempMask[idx]) return make_float3(0.0f);
	return state.d_pAlbedo[p00];
}

__device__ void applyJpAlbedSpatialDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);


	if (p00 < 0 || !state.d_optimizeMask[idx]) {
		state.d_Jp_albedo_spatial[4 * idx + 0] = make_float3(0);
		state.d_Jp_albedo_spatial[4 * idx + 1] = make_float3(0);
		state.d_Jp_albedo_spatial[4 * idx + 2] = make_float3(0);
		state.d_Jp_albedo_spatial[4 * idx + 3] = make_float3(0);
		return;
	}

	float3 p_p00 = state.d_pAlbedo[p00];

	state.d_Jp_albedo_spatial[4 * p00 + 0] = sqrtf(chromacityWeight(p00, pn0, state)) * (p_p00 - state.d_pAlbedo[pn0]);
	state.d_Jp_albedo_spatial[4 * p00 + 1] = sqrtf(chromacityWeight(p00, p0n, state)) * (p_p00 - state.d_pAlbedo[p0n]);
	state.d_Jp_albedo_spatial[4 * p00 + 2] = sqrtf(chromacityWeight(p00, pp0, state)) * (p_p00 - state.d_pAlbedo[pp0]);
	state.d_Jp_albedo_spatial[4 * p00 + 3] = sqrtf(chromacityWeight(p00, p0p, state)) * (p_p00 - state.d_pAlbedo[p0p]);
}

__device__ float applyJTJpShading3DeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams) {
	// Normal orientation, we have to minus x and z.
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0) return 0;

	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
	int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
	int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);
	int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);

	float x = p002D.x;
	float y = p002D.y;
	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;
	float resDepth = 0.0f;

	// Our target is D(i, j) which is p00.

#ifdef CENTER_NORMAL
	// Eg(i - 1, j)
	// D(i, j) is in || B(i - 1, j) - I(i - 1, j) ||
	// dB(i + 1, j) / dD(i, j)
	if (pn0 >= 0 && state.d_optimizeMask[pn0]) {
		float3 albedon0 = make_float3(state.d_updateAlbedo[pn0]);
		float3 cam_nn0 = NormalFromDepth(pn0, state, enhanceParams);
		float3 nn0 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_nn0, 0.0f));
		float3 dBn0dNn0_r = albedon0.x * dLightdNormal(state.d_updateLight, nn0);
		float3 dBn0dNn0_g = albedon0.y * dLightdNormal(state.d_updateLight, nn0);
		float3 dBn0dNn0_b = albedon0.z * dLightdNormal(state.d_updateLight, nn0);

		float dNn0dD00_x = (state.d_updateDepth[pnn] + state.d_updateDepth[pnp]) / fy;
		float dNn0dD00_y = (state.d_updateDepth[pnp] - state.d_updateDepth[pnn]) / fx;
		float dNn0dD00_z = ((dNn0dD00_x * (cx - (x - 1.0f)) / fx) + (dNn0dD00_y * (cy - y) / fy) - ((state.d_updateDepth[pnn] + state.d_updateDepth[pnp]) / fx / fy));
		float3 dNn0dD00 = dUNormaldNormal(pn0, state, enhanceParams) * make_float3(dNn0dD00_x, dNn0dD00_y, dNn0dD00_z);
		float3 dBn0dD00 = make_float3(dot(dBn0dNn0_r, dNn0dD00), dot(dBn0dNn0_g, dNn0dD00), dot(dBn0dNn0_b, dNn0dD00));
		float dBn0dD00_dot_Jpn0 = dot(dBn0dD00, state.d_Jp_shading[pn0]);
		resDepth += dBn0dD00_dot_Jpn0;
	}

	// Eg(i, j - 1)
	// D(i, j) is in || B(i, j - 1) - I(i, j - 1) ||
	// dB(i, j - 1) / dD(i, j)
	if (p0n >= 0 && state.d_optimizeMask[p0n]) {
		float3 albedo0n = make_float3(state.d_updateAlbedo[p0n]);
		float3 cam_n0n = NormalFromDepth(p0n, state, enhanceParams);
		float3 n0n = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n0n, 0.0f));
		float3 dB0ndN0n_r = albedo0n.x * dLightdNormal(state.d_updateLight, n0n);
		float3 dB0ndN0n_g = albedo0n.y * dLightdNormal(state.d_updateLight, n0n);
		float3 dB0ndN0n_b = albedo0n.z * dLightdNormal(state.d_updateLight, n0n);

		float dN0ndD00_x = (state.d_updateDepth[ppn] - state.d_updateDepth[pnn]) / fy;
		float dN0ndD00_y = (state.d_updateDepth[pnn] + state.d_updateDepth[ppn]) / fx;
		float dN0ndD00_z = ((dN0ndD00_x * (cx - x) / fx) + (dN0ndD00_y * (cy - (y - 1.0f)) / fy) - ((state.d_updateDepth[pnn] + state.d_updateDepth[ppn]) / fx / fy));
		float3 dN0ndD00 = dUNormaldNormal(p0n, state, enhanceParams) * make_float3(dN0ndD00_x, dN0ndD00_y, dN0ndD00_z);
		float3 dB0ndD00 = make_float3(dot(dB0ndN0n_r, dN0ndD00), dot(dB0ndN0n_g, dN0ndD00), dot(dB0ndN0n_b, dN0ndD00));
		float dB0ndD00_dot_Jp0n = dot(dB0ndD00, state.d_Jp_shading[p0n]);
		resDepth += dB0ndD00_dot_Jp0n;
	}

	// Eg(i + 1, j)
	// D(i, j) is in || B(i + 1, j) - I(i + 1, j) ||
	// dB(i + 1, j) / dD(i, j)
	if (pp0 >= 0 && state.d_optimizeMask[pp0]) {
		float3 albedop0 = make_float3(state.d_updateAlbedo[pp0]);
		float3 cam_np0 = NormalFromDepth(pp0, state, enhanceParams);
		float3 np0 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_np0, 0.0f));
		float3 dBp0dNp0_r = albedop0.x * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_g = albedop0.y * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_b = albedop0.z * dLightdNormal(state.d_updateLight, np0);

		float dNp0dD00_x = -(state.d_updateDepth[ppn] + state.d_updateDepth[ppp]) / fy;
		float dNp0dD00_y = (state.d_updateDepth[ppp] - state.d_updateDepth[ppn]) / fx;
		float dNp0dD00_z = ((dNp0dD00_x * (cx - (x + 1.0f)) / fx) + (dNp0dD00_y * (cy - y) / fy) - ((state.d_updateDepth[ppn] + state.d_updateDepth[ppp]) / fx / fy));
		float3 dNp0dD00 = dUNormaldNormal(pp0, state, enhanceParams) * make_float3(dNp0dD00_x, dNp0dD00_y, dNp0dD00_z);
		float3 dBp0dD00 = make_float3(dot(dBp0dNp0_r, dNp0dD00), dot(dBp0dNp0_g, dNp0dD00), dot(dBp0dNp0_b, dNp0dD00));
		float dBp0dD00_dot_Jpp0 = dot(dBp0dD00, state.d_Jp_shading[pp0]);
		resDepth += dBp0dD00_dot_Jpp0;
	}

	// Eg(i, j + 1)
	// D(i, j) is in || B(i, j + 1) - I(i, j + 1) ||
	// dB(i, j + 1) / dD(i, j)
	if (p0p >= 0 && state.d_optimizeMask[p0p]) {
		float3 albedo0p = make_float3(state.d_updateAlbedo[p0p]);
		float3 cam_n0p = NormalFromDepth(p0p, state, enhanceParams);
		float3 n0p = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n0p, 0.0f));
		float3 dB0pdN0p_r = albedo0p.x * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_g = albedo0p.y * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_b = albedo0p.z * dLightdNormal(state.d_updateLight, n0p);

		float dN0pdD00_x = (state.d_updateDepth[ppp] - state.d_updateDepth[pnp]) / fy;
		float dN0pdD00_y = -(state.d_updateDepth[pnp] + state.d_updateDepth[ppp]) / fx;
		float dN0pdD00_z = ((dN0pdD00_x * (cx - x) / fx) + (dN0pdD00_y * (cy - (y + 1.0f)) / fy) - ((state.d_updateDepth[pnp] + state.d_updateDepth[ppp]) / fx / fy));
		float3 dN0pdD00 = dUNormaldNormal(p0p, state, enhanceParams) * make_float3(dN0pdD00_x, dN0pdD00_y, dN0pdD00_z);
		float3 dB0pdD00 = make_float3(dot(dB0pdN0p_r, dN0pdD00), dot(dB0pdN0p_g, dN0pdD00), dot(dB0pdN0p_b, dN0pdD00));
		float dB0pdD00_dot_Jp0p = dot(dB0pdD00, state.d_Jp_shading[p0p]);
		resDepth += dB0pdD00_dot_Jp0p;
	}
#else
	// Eg(i, j)
	// D(i, j) is in || B(i, j) - I(i, j) ||
	if (state.d_iterMask[p00]) {
		float3 albedo00 = state.d_updateAlbedo[p00];
		float3 n00 = NormalFromDepth(p00, state, enhanceParams);
		float3 dB00dN00_r = albedo00.x * dLightdNormal(state.d_updateLight, n00);
		float3 dB00dN00_g = albedo00.y * dLightdNormal(state.d_updateLight, n00);
		float3 dB00dN00_b = albedo00.z * dLightdNormal(state.d_updateLight, n00);

		// dB(i, j) / dD(i, j)
		float dN00dD00_x = (state.d_updateDepth[p0n] / fy);
		float dN00dD00_y = state.d_updateDepth[pn0] / fx;
		float dN00dD00_z = (dN00dD00_x * (cx - x) / fx + dN00dD00_y * (cy - y) / fy);
		float3 dN00dD00 = dUNormaldNormal(p00, state, enhanceParams) * make_float3(dN00dD00_x, dN00dD00_y, dN00dD00_z);
		float3 dB00dD00 = make_float3(dot(dB00dN00_r, dN00dD00), dot(dB00dN00_g, dN00dD00), dot(dB00dN00_b, dN00dD00));
		float dB00dD00_dot_Jp00 = dot(dB00dD00, state.d_Jp_shading[p00]);
		resDepth += dB00dD00_dot_Jp00;
	}

	// Eg(i, j + 1)
	// D(i, j) is in || B(i, j + 1) - I(i, j + 1) ||
	// dB(i, j + 1) / dD(i, j)
	if (p0p >= 0 && state.d_iterMask[p0p]) {
		float3 albedo0p = state.d_updateAlbedo[p0p];
		float3 n0p = NormalFromDepth(p0p, state, enhanceParams);
		float3 dB0pdN0p_r = albedo0p.x * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_g = albedo0p.y * dLightdNormal(state.d_updateLight, n0p);
		float3 dB0pdN0p_b = albedo0p.z * dLightdNormal(state.d_updateLight, n0p);

		float dN0pdD00_x = (state.d_updateDepth[p0p] - state.d_updateDepth[pnp]) / fy;
		float dN0pdD00_y = -state.d_updateDepth[pnp] / fx;
		float dN0pdD00_z = (dN0pdD00_x * (cx - x) / fx + dN0pdD00_y * (cy - (y + 1.0f)) / fy - state.d_updateDepth[pnp] / fx / fy);
		float3 dN0pdD00 = dUNormaldNormal(p0p, state, enhanceParams) * make_float3(dN0pdD00_x, dN0pdD00_y, dN0pdD00_z);
		float3 dB0pdD00 = make_float3(dot(dB0pdN0p_r, dN0pdD00), dot(dB0pdN0p_g, dN0pdD00), dot(dB0pdN0p_b, dN0pdD00));
		float dB0pdD00_dot_Jp0p = dot(dB0pdD00, state.d_Jp_shading[p0p]);
		resDepth += dB0pdD00_dot_Jp0p;
	}

	// Eg(i + 1, j)
	// D(i, j) is in || B(i + 1, j) - I(i + 1, j) ||
	// dB(i + 1, j) / dD(i, j)
	if (pp0 >= 0 && state.d_iterMask[pp0]) {
		float3 albedop0 = state.d_updateAlbedo[pp0];
		float3 np0 = NormalFromDepth(pp0, state, enhanceParams);
		float3 dBp0dNp0_r = albedop0.x * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_g = albedop0.y * dLightdNormal(state.d_updateLight, np0);
		float3 dBp0dNp0_b = albedop0.z * dLightdNormal(state.d_updateLight, np0);

		float dNp0dD00_x = -1.0f * state.d_updateDepth[ppn] / fy;
		float dNp0dD00_y = (state.d_updateDepth[pp0] - state.d_updateDepth[ppn]) / fx;
		float dNp0dD00_z = (dNp0dD00_x * (cx - (x + 1.0f)) / fx + dNp0dD00_y * (cy - y) / fy - state.d_updateDepth[ppn] / fx / fy);
		float3 dNp0dD00 = dUNormaldNormal(pp0, state, enhanceParams) * make_float3(dNp0dD00_x, dNp0dD00_y, dNp0dD00_z);
		float3 dB0pdD00 = make_float3(dot(dBp0dNp0_r, dNp0dD00), dot(dBp0dNp0_g, dNp0dD00), dot(dBp0dNp0_b, dNp0dD00));
		float dBp0dD00_dot_Jpp0 = dot(dB0pdD00, state.d_Jp_shading[pp0]);
		resDepth += dBp0dD00_dot_Jpp0;
	}
#endif
	return resDepth;
}


__device__ float applyJTJpRegDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) return 0.0f;
	return state.d_pDepth[p00];
}

__device__ float applyJTJpSmoothDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
	int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
	int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);
	int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
	int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);

	if (p00 < 0) return 0.0f;
	float dJtJp_smooth = 0.0f;

	float weight = -0.25f;
#ifdef LAPLACIAN_OCTA
	weight = -0.125f;
#endif

	if (state.d_optimizeMask[p00]) 				dJtJp_smooth += state.d_Jp_smooth[p00];
	if (pn0 >= 0 && state.d_optimizeMask[pn0])	dJtJp_smooth += weight * state.d_Jp_smooth[pn0];
	if (p0n >= 0 && state.d_optimizeMask[p0n])	dJtJp_smooth += weight * state.d_Jp_smooth[p0n];
	if (pp0 >= 0 && state.d_optimizeMask[pp0])	dJtJp_smooth += weight * state.d_Jp_smooth[pp0];
	if (p0p >= 0 && state.d_optimizeMask[p0p])	dJtJp_smooth += weight * state.d_Jp_smooth[p0p];
#ifdef LAPLACIAN_OCTA
	if (ppn >= 0 && state.d_optimizeMask[ppn])	dJtJp_smooth += weight * state.d_Jp_smooth[ppn];
	if (ppp >= 0 && state.d_optimizeMask[ppp])	dJtJp_smooth += weight * state.d_Jp_smooth[ppp];
	if (pnp >= 0 && state.d_optimizeMask[pnp])	dJtJp_smooth += weight * state.d_Jp_smooth[pnp];
	if (pnn >= 0 && state.d_optimizeMask[pnn])	dJtJp_smooth += weight * state.d_Jp_smooth[pnn];
#endif
	return dJtJp_smooth;
}

__device__ float3 applyJTJpShadingAlbedoDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams, RayCastParams& rayCastParams) {
	// Normal orientation, we have to minus x and z.
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx]) return make_float3(0.0f);

	float3 resAlbedo = make_float3(0.0f);

	float3 cam_n00 = NormalFromDepth(p00, state, enhanceParams);
	float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));

	float3 dB00dA00 = make_float3(computeLightKernel(state.d_updateLight, n00));
	float3 dB00dA00_Jp00 = dB00dA00 * state.d_Jp_shading[p00];

	resAlbedo += dB00dA00_Jp00;

	return resAlbedo;
}

__device__ float3 applyJTJpAlbedoTempDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);

	if (p00 < 0 || !state.d_optimizeMask[idx] || !state.d_albedoTempMask[idx]) return make_float3(0.0f);
	//return state.d_Jp_albedo_temp[p00];
	return state.d_pAlbedo[p00];
}

__device__ float3 applyJTJpAlbedoSpatialDeviceBruteForce(int idx, InputEnhanceData& state, InputEnhanceParams& enhanceParams) {
	// Normal orientation, we have to minus x and z.
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = getPixel1DCoordinate(p002D, state, enhanceParams);
	int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
	int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
	int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);

	if (p00 < 0) return make_float3(0.0f);

	float3 resAlbedo = make_float3(0.0f);

	if (state.d_optimizeMask[p00]) {
		resAlbedo += sqrtf(chromacityWeight(p00, pn0, state)) * state.d_Jp_albedo_spatial[4 * p00 + 0];
		resAlbedo += sqrtf(chromacityWeight(p00, p0n, state)) * state.d_Jp_albedo_spatial[4 * p00 + 1];
		resAlbedo += sqrtf(chromacityWeight(p00, pp0, state)) * state.d_Jp_albedo_spatial[4 * p00 + 2];
		resAlbedo += sqrtf(chromacityWeight(p00, p0p, state)) * state.d_Jp_albedo_spatial[4 * p00 + 3];
	}

	if (pn0 >= 0 && state.d_optimizeMask[pn0]) resAlbedo += -sqrtf(chromacityWeight(pn0, p00, state)) * state.d_Jp_albedo_spatial[4 * pn0 + 2];
	if (p0n >= 0 && state.d_optimizeMask[p0n]) resAlbedo += -sqrtf(chromacityWeight(p0n, p00, state)) * state.d_Jp_albedo_spatial[4 * p0n + 3];
	if (pp0 >= 0 && state.d_optimizeMask[pp0]) resAlbedo += -sqrtf(chromacityWeight(pp0, p00, state)) * state.d_Jp_albedo_spatial[4 * pp0 + 0];
	if (p0p >= 0 && state.d_optimizeMask[p0p]) resAlbedo += -sqrtf(chromacityWeight(p0p, p00, state)) * state.d_Jp_albedo_spatial[4 * p0p + 1];

	return resAlbedo;
}


/////////////////////////////////////////////////////////////////////////
// PCG Evaluation Parts
/////////////////////////////////////////////////////////////////////////

__global__ void EvalResidualDevice(InputEnhanceData state, InputEnhanceParams enhanceParams, RayCastParams rayCastParams)
{
	const unsigned int N = enhanceParams.nImagePixel;					// Number of block variables
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float residual_data_shading = 0.0f;
	float residual_reg = 0.0f;
	float residual_smooth = 0.0f;
	float residual_albedo_temp = 0.0f;
	float residual_albedo_spatial = 0.0f;
	if (idx >= 0 && idx < N && state.d_optimizeMask[idx])
	{
		int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
		int p00 = idx;
		int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
		int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
		int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
		int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
		int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
		int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);
		int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
		int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);

		if (pp0 < 0 || p0p < 0 || pn0 < 0 || p0n < 0 || ppn < 0 || ppp < 0 || pnp < 0 || pnn < 0)
			return;

		// Eg(i, j) = || B(i, j) - B(i + 1, j) - (I(i, j) - I(i + 1, j)) ||
		float3 albedo00 = make_float3(state.d_srcAlbedo[p00] + make_float4(state.d_deltaAlbedo[p00], 0.0f));// (state.d_deltaAlbedo[p00].x + state.d_deltaAlbedo[p00].y + state.d_deltaAlbedo[p00].z) / 3.0f + state.d_srcGrayAlbedo[p00];
		float3 cam_n00 = NormalFromDeltaDepth(p00, state, enhanceParams);
		float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));
		if (NormalLengthFromDeltaDepth(p00, state, enhanceParams) == 0) return;
		float3 B00 = albedo00 * computeDeltaLightKernel(state.d_updateLight, state.d_deltaLight, n00);

		float3 F00_shading = (B00 - make_float3(state.d_srcImage[p00]));

		residual_data_shading = dot(F00_shading, F00_shading);

		float iterD00 = state.d_updateDepth[p00] + state.d_deltaDepth[p00];
		float iterDn0 = state.d_updateDepth[pn0] + state.d_deltaDepth[pn0];
		float iterD0n = state.d_updateDepth[p0n] + state.d_deltaDepth[p0n];
		float iterDp0 = state.d_updateDepth[pp0] + state.d_deltaDepth[pp0];
		float iterD0p = state.d_updateDepth[p0p] + state.d_deltaDepth[p0p];

		residual_reg = (state.d_srcDepth[p00] - iterD00) * (state.d_srcDepth[p00] - iterD00);

#ifdef AVERAGE_SMOOTHNESS_TERM
#ifdef LAPLACIAN_OCTA
		float iterDpn = state.d_updateDepth[ppn] + state.d_deltaDepth[ppn];
		float iterDpp = state.d_updateDepth[ppp] + state.d_deltaDepth[ppp];
		float iterDnp = state.d_updateDepth[pnp] + state.d_deltaDepth[pnp];
		float iterDnn = state.d_updateDepth[pnn] + state.d_deltaDepth[pnn];
		residual_smooth = (iterD00 - 0.125f * (iterDn0 + iterD0n + iterDp0 + iterD0p + iterDpn + iterDpp + iterDnp + iterDnn))
			* (iterD00 - 0.125f * (iterDn0 + iterD0n + iterDp0 + iterD0p + iterDpn + iterDpp + iterDnp + iterDnn));
#else
		residual_smooth = (iterD00 - 0.25f * (iterDn0 + iterD0n + iterDp0 + iterD0p)) * (iterD00 - 0.25f * (iterDn0 + iterD0n + iterDp0 + iterD0p));
#endif
#else
		residual_smooth = (iterD00 - iterDn0) * (iterD00 - iterDn0);
		residual_smooth += (iterD00 - iterD0n) * (iterD00 - iterD0n);
		residual_smooth += (iterD00 - iterDp0) * (iterD00 - iterDp0);
		residual_smooth += (iterD00 - iterD0p) * (iterD00 - iterD0p);
#endif

		if (enhanceParams.optimizeAlbedo) {
			float3 albedon0 = make_float3(state.d_updateAlbedo[pn0] + make_float4(state.d_deltaAlbedo[pn0], 0.0f));
			float3 albedo0n = make_float3(state.d_updateAlbedo[p0n] + make_float4(state.d_deltaAlbedo[p0n], 0.0f));
			float3 albedop0 = make_float3(state.d_updateAlbedo[pp0] + make_float4(state.d_deltaAlbedo[pp0], 0.0f));
			float3 albedo0p = make_float3(state.d_updateAlbedo[p0p] + make_float4(state.d_deltaAlbedo[p0p], 0.0f));

			float3 F00_n0_albedo_spatial = albedo00 - albedon0;
			float3 F00_0n_albedo_spatial = albedo00 - albedo0n;
			float3 F00_p0_albedo_spatial = albedo00 - albedop0;
			float3 F00_0p_albedo_spatial = albedo00 - albedo0p;
			residual_albedo_spatial = chromacityWeight(p00, pn0, state) * dot(F00_n0_albedo_spatial, F00_n0_albedo_spatial) +
				chromacityWeight(p00, p0n, state) * dot(F00_0n_albedo_spatial, F00_0n_albedo_spatial) +
				chromacityWeight(p00, pp0, state) * dot(F00_p0_albedo_spatial, F00_p0_albedo_spatial) +
				chromacityWeight(p00, p0p, state) * dot(F00_0p_albedo_spatial, F00_0p_albedo_spatial);

			if (state.d_albedoTempMask[p00]) {
				float3 albedoDiff = albedo00 - make_float3(state.d_srcAlbedo[p00]);
				residual_albedo_temp = dot(albedoDiff, albedoDiff);
			}
		}

		state.d_debugResidualData[idx] = enhanceParams.weightDataShading * residual_data_shading +
			enhanceParams.weightDepthTemp * residual_reg +
			enhanceParams.weightDepthSmooth * residual_smooth +
			enhanceParams.weightAlbedoTemp * residual_albedo_temp +
			enhanceParams.weightAlbedoSmooth * residual_albedo_spatial;

		residual_data_shading = warpReduce(residual_data_shading);
		residual_reg = warpReduce(residual_reg);
		residual_smooth = warpReduce(residual_smooth);
		residual_albedo_temp = warpReduce(residual_albedo_temp);
		residual_albedo_spatial = warpReduce(residual_albedo_spatial);

		if (threadIdx.x % WARP_SIZE == 0) {
			atomicAdd(state.d_sumResidual, enhanceParams.weightDataShading * residual_data_shading);
			atomicAdd(&state.d_sumResidual[3], enhanceParams.weightDepthTemp * residual_reg);
			atomicAdd(&state.d_sumResidual[4], enhanceParams.weightDepthSmooth * residual_smooth);
			atomicAdd(&state.d_sumResidual[6], enhanceParams.weightAlbedoTemp * residual_albedo_temp);
			atomicAdd(&state.d_sumResidual[7], enhanceParams.weightAlbedoSmooth * residual_albedo_spatial);
		}
	}
}

__global__ void EvalInitialResidualDevice(InputEnhanceData state, InputEnhanceParams enhanceParams, RayCastParams rayCastParams)
{
	const unsigned int N = enhanceParams.nImagePixel;					// Number of block variables
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float residual_data_shading = 0.0f;
	float residual_reg = 0.0f;
	float residual_smooth = 0.0f;
	float residual_albedo_temp = 0.0f;
	float residual_albedo_spatial = 0.0f;

	if (idx >= 0 && idx < N && state.d_srcMask[idx])
	{
		int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
		int p00 = idx;
		int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), state, enhanceParams);
		int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), state, enhanceParams);
		int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), state, enhanceParams);
		int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), state, enhanceParams);
		int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), state, enhanceParams);
		int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), state, enhanceParams);
		int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), state, enhanceParams);
		int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), state, enhanceParams);

		if (pp0 < 0 || p0p < 0 || pn0 < 0 || p0n < 0 || ppn < 0 || ppp < 0 || pnp < 0 || pnn < 0)
			return;

		// Eg(i, j) = || B(i, j) - B(i + 1, j) - (I(i, j) - I(i + 1, j)) ||
		float3 albedo00 = make_float3(state.d_updateAlbedo[p00]);
		float3 cam_n00 = NormalFromDepth(p00, state, enhanceParams);
		float3 n00 = make_float3(rayCastParams.m_viewMatrixInverse * make_float4(cam_n00, 0.0f));
		if (NormalLengthFromDepth(p00, state, enhanceParams) == 0) return;
		float3 B00 = albedo00 * computeLightKernel(state.d_updateLight, n00);

		float3 F00_shading = B00 - make_float3(state.d_srcImage[p00]);
		residual_data_shading = dot(F00_shading, F00_shading);

		float iterD00 = state.d_updateDepth[p00];
		float iterDn0 = state.d_updateDepth[pn0];
		float iterD0n = state.d_updateDepth[p0n];
		float iterDp0 = state.d_updateDepth[pp0];
		float iterD0p = state.d_updateDepth[p0p];

		residual_reg = (state.d_srcDepth[p00] - iterD00) * (state.d_srcDepth[p00] - iterD00);

#ifdef AVERAGE_SMOOTHNESS_TERM
#ifdef LAPLACIAN_OCTA
		float iterDpn = state.d_updateDepth[ppn];
		float iterDpp = state.d_updateDepth[ppp];
		float iterDnp = state.d_updateDepth[pnp];
		float iterDnn = state.d_updateDepth[pnn];
		residual_smooth = (iterD00 - 0.125f * (iterDn0 + iterD0n + iterDp0 + iterD0p + iterDpn + iterDpp + iterDnp + iterDnn))
			* (iterD00 - 0.125f * (iterDn0 + iterD0n + iterDp0 + iterD0p + iterDpn + iterDpp + iterDnp + iterDnn));
#else
		residual_smooth = (iterD00 - 0.25f * (iterDn0 + iterD0n + iterDp0 + iterD0p)) * (iterD00 - 0.25f * (iterDn0 + iterD0n + iterDp0 + iterD0p));
#endif
#else
		residual_smooth = (iterD00 - iterDn0) * (iterD00 - iterDn0);
		residual_smooth += (iterD00 - iterD0n) * (iterD00 - iterD0n);
		residual_smooth += (iterD00 - iterDp0) * (iterD00 - iterDp0);
		residual_smooth += (iterD00 - iterD0p) * (iterD00 - iterD0p);
#endif

		if (enhanceParams.optimizeAlbedo) {
			float3 albedon0 = make_float3(state.d_updateAlbedo[pn0]);
			float3 albedo0n = make_float3(state.d_updateAlbedo[p0n]);
			float3 albedop0 = make_float3(state.d_updateAlbedo[pp0]);
			float3 albedo0p = make_float3(state.d_updateAlbedo[p0p]);
			float3 F00_n0_albedo_spatial = albedo00 - albedon0;
			float3 F00_0n_albedo_spatial = albedo00 - albedo0n;
			float3 F00_p0_albedo_spatial = albedo00 - albedop0;
			float3 F00_0p_albedo_spatial = albedo00 - albedo0p;
			residual_albedo_spatial = chromacityWeight(p00, pn0, state) * dot(F00_n0_albedo_spatial, F00_n0_albedo_spatial) +
				chromacityWeight(p00, p0n, state) * dot(F00_0n_albedo_spatial, F00_0n_albedo_spatial) +
				chromacityWeight(p00, pp0, state) * dot(F00_p0_albedo_spatial, F00_p0_albedo_spatial) +
				chromacityWeight(p00, p0p, state) * dot(F00_0p_albedo_spatial, F00_0p_albedo_spatial);
			if (state.d_albedoTempMask[p00]) {
				float3 albedoDiff = albedo00 - make_float3(state.d_srcAlbedo[p00]);
				residual_albedo_temp = dot(albedoDiff, albedoDiff);
			}
		}

		state.d_debugResidualData[idx] = enhanceParams.weightDataShading * residual_data_shading +
			enhanceParams.weightDepthTemp * residual_reg +
			enhanceParams.weightDepthSmooth * residual_smooth +
			enhanceParams.weightAlbedoTemp * residual_albedo_temp +
			enhanceParams.weightAlbedoSmooth * residual_albedo_spatial;

		residual_data_shading = warpReduce(residual_data_shading);
		residual_reg = warpReduce(residual_reg);
		residual_smooth = warpReduce(residual_smooth);
		residual_albedo_temp = warpReduce(residual_albedo_temp);
		residual_albedo_spatial = warpReduce(residual_albedo_spatial);

		if (threadIdx.x % WARP_SIZE == 0) {
			atomicAdd(state.d_sumResidual, enhanceParams.weightDataShading * residual_data_shading);
			atomicAdd(&state.d_sumResidual[3], enhanceParams.weightDepthTemp * residual_reg);
			atomicAdd(&state.d_sumResidual[4], enhanceParams.weightDepthSmooth * residual_smooth);
			atomicAdd(&state.d_sumResidual[6], enhanceParams.weightAlbedoTemp * residual_albedo_temp);
			atomicAdd(&state.d_sumResidual[7], enhanceParams.weightAlbedoSmooth * residual_albedo_spatial);
		}
	}
}


/////////////////////////////////////////////////////////////////////////
// PCG Initialization Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGInit_Kernel1(InputEnhanceData state, InputEnhanceParams enhanceParams, RayCastParams rayCastParams)
{
	const unsigned int N = enhanceParams.nImagePixel;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x >= 0 && x < N)
	{
		float resDepth = 0.0f;
		float preDepth = 0.0f;
		float3 resAlbedo = make_float3(0);
		float3 preAlbedo = make_float3(0);
		state.d_deltaDepth[x] = 0.0f;
		state.d_preconditionerDepth[x] = 1.0f;
		state.d_deltaAlbedo[x] = make_float3(0);
		state.d_preconditionerAlbedo[x] = make_float3(1.0f);

		float pDepth = 0.0f;
		if (enhanceParams.optimizeDepth) {
			evalMinusJTFDeviceShading3(x, state, enhanceParams, rayCastParams, resDepth, preDepth);
			evalMinusJTFDeviceReg(x, state, enhanceParams, resDepth, preDepth);
			evalMinusJTFDeviceSmooth(x, state, enhanceParams, resDepth, preDepth);

#ifdef PRECONTIDIONER_APPLY
			if (preDepth > FLOAT_EPSILON)	state.d_preconditionerDepth[x] = 1.0f / preDepth;
			else							state.d_preconditionerDepth[x] = 1.0f;
#endif
			pDepth = state.d_preconditionerDepth[x] * resDepth;			// apply preconditioner M^-1
		}
		state.d_rDepth[x] = resDepth;										// store for next iteration
		state.d_pDepth[x] = pDepth;

		float3 pAlbedo = make_float3(0);
		if (enhanceParams.optimizeAlbedo) {
			evalMinusJTFDeviceShadingAlbedo(x, state, enhanceParams, rayCastParams, resAlbedo, preAlbedo);
			evalMinusJTFDeviceAlbedoTemp(x, state, enhanceParams, resAlbedo, preAlbedo);
			evalMinusJTFDeviceAlbedoSpatial(x, state, enhanceParams, resAlbedo, preAlbedo);

#ifdef PRECONTIDIONER_APPLY
			if (preAlbedo.x > FLOAT_EPSILON)	state.d_preconditionerAlbedo[x].x = 1.0f / preAlbedo.x;
			else								state.d_preconditionerAlbedo[x].x = 1.0f;
			if (preAlbedo.y > FLOAT_EPSILON)	state.d_preconditionerAlbedo[x].y = 1.0f / preAlbedo.y;
			else								state.d_preconditionerAlbedo[x].y = 1.0f;
			if (preAlbedo.z > FLOAT_EPSILON)	state.d_preconditionerAlbedo[x].z = 1.0f / preAlbedo.z;
			else								state.d_preconditionerAlbedo[x].z = 1.0f;
#endif
			pAlbedo = state.d_preconditionerAlbedo[x] * resAlbedo;			// apply preconditioner M^-1

		}

		state.d_rAlbedo[x] = resAlbedo;
		state.d_pAlbedo[x] = pAlbedo;

		d = resDepth * pDepth + dot(resAlbedo, pAlbedo);					// x-th term of nomimator for computing alpha and denominator for computing beta

		state.d_Ap_Depth[x] = 0.0f;
		state.d_Ap_Albedo[x] = make_float3(0.0f);
	}

	d = warpReduce(d);

	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGInit_Kernel2(unsigned int N, InputEnhanceData state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];				// store result for next kernel call
}


/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel0(InputEnhanceData state, InputEnhanceParams enhanceParams, RayCastParams rayCastParams)
{
	const unsigned int N = enhanceParams.nImagePixel;					// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= 0 && x < N)
	{
		float3 tmp0 = make_float3(0.0f);

		if (enhanceParams.optimizeDepth) {
			tmp0 = applyJpShading3DeviceBruteForce(x, state, enhanceParams, rayCastParams);
			applyJpSmoothDeviceBruteForce(x, state, enhanceParams); // A x p_k  => J^T x J x p_k 
		}

		if (enhanceParams.optimizeAlbedo) {
			tmp0 += applyJpShadingAlbedoDeviceBruteForce(x, state, enhanceParams, rayCastParams);
			applyJpAlbedSpatialDeviceBruteForce(x, state, enhanceParams);
		}

		state.d_Jp_shading[x] = tmp0;
	}
}

__global__ void PCGStep_Kernel1a(InputEnhanceData state, InputEnhanceParams enhanceParams, RayCastParams rayCastParams)
{
	const unsigned int N = enhanceParams.nImagePixel;							// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= 0 && x < N)
	{
		if (enhanceParams.optimizeDepth) {
			const float shading = applyJTJpShading3DeviceBruteForce(x, state, enhanceParams, rayCastParams);

			const float reg = applyJTJpRegDeviceBruteForce(x, state, enhanceParams);
			const float smooth = applyJTJpSmoothDeviceBruteForce(x, state, enhanceParams);

			state.d_Ap_Depth[x] = enhanceParams.weightDataShading * shading +
				enhanceParams.weightDepthTemp * reg +
				enhanceParams.weightDepthSmooth * smooth;
		}

		if (enhanceParams.optimizeAlbedo) {
			const float3 shadingAlbedo = applyJTJpShadingAlbedoDeviceBruteForce(x, state, enhanceParams, rayCastParams);
			const float3 albedoTemp = applyJTJpAlbedoTempDeviceBruteForce(x, state, enhanceParams);
			const float3 albedoSpatial = applyJTJpAlbedoSpatialDeviceBruteForce(x, state, enhanceParams);
			state.d_Ap_Albedo[x] = enhanceParams.weightDataShading * shadingAlbedo +
				enhanceParams.weightAlbedoTemp * albedoTemp +
				enhanceParams.weightAlbedoSmooth * albedoSpatial;
		}

	}
}

__global__ void PCGStep_Kernel1b(InputEnhanceData state, InputEnhanceParams enhanceParams)
{
	const unsigned int N = enhanceParams.nImagePixel;								// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x >= 0 && x < N)
	{
		d = state.d_pDepth[x] * state.d_Ap_Depth[x] + dot(state.d_pAlbedo[x], state.d_Ap_Albedo[x]); // x-th term of denominator of alpha
	}

	d = warpReduce(d);

	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGStep_Kernel2(InputEnhanceData state, InputEnhanceParams enhanceParams)
{
	const unsigned int N = enhanceParams.nImagePixel;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x >= 0 && x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;		// update step size alpha

		state.d_deltaDepth[x] = state.d_deltaDepth[x] + alpha * state.d_pDepth[x];		// do a decent step
		state.d_deltaAlbedo[x] = state.d_deltaAlbedo[x] + alpha * state.d_pAlbedo[x];	// do a decent step

		float rDepth = state.d_rDepth[x] - alpha * state.d_Ap_Depth[x];					// update residuum
		state.d_rDepth[x] = rDepth;														// store for next kernel call

		float3 rAlbedo = state.d_rAlbedo[x] - alpha * state.d_Ap_Albedo[x];				// update residuum
		state.d_rAlbedo[x] = rAlbedo;													// store for next kernel call

		float zDepth = state.d_preconditionerDepth[x] * rDepth;							// apply preconditioner M^-1
		state.d_zDepth[x] = zDepth;														// save for next kernel call

		float3 zAlbedo = state.d_preconditionerAlbedo[x] * rAlbedo;						//   apply preconditioner M^-1
		state.d_zAlbedo[x] = zAlbedo;													// save for next kernel call
		
		b = zDepth * rDepth + dot(zAlbedo, rAlbedo);									// compute x-th term of the nominator of beta
	}

	b = warpReduce(b);

	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(&state.d_scanAlpha[1], b);
	}
}

template<bool lastIteration>
__global__ void PCGStep_Kernel3(InputEnhanceData state, InputEnhanceParams enhanceParams)
{
	const unsigned int N = enhanceParams.nImagePixel;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= 0 && x < N)
	{
		const float rDotzNew = state.d_scanAlpha[1];								// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;				// update step size beta

		state.d_rDotzOld[x] = rDotzNew;											// save new rDotz for next iteration
		state.d_pDepth[x] = state.d_zDepth[x] + beta * state.d_pDepth[x];		// update decent direction
		state.d_pAlbedo[x] = state.d_zAlbedo[x] + beta * state.d_pAlbedo[x];	// update decent direction

		state.d_Ap_Depth[x] = 0.0f;
		state.d_Ap_Albedo[x] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{
			state.d_updateDepth[x] = fmaxf(0.0f, state.d_updateDepth[x] + state.d_deltaDepth[x]);
			state.d_updateAlbedo[x].x = fmaxf(0.000001f, (state.d_updateAlbedo[x].x + state.d_deltaAlbedo[x].x));
			state.d_updateAlbedo[x].y = fmaxf(0.000001f, (state.d_updateAlbedo[x].y + state.d_deltaAlbedo[x].y));
			state.d_updateAlbedo[x].z = fmaxf(0.000001f, (state.d_updateAlbedo[x].z + state.d_deltaAlbedo[x].z));
			if (x < enhanceParams.nLightCoefficient) {
				state.d_updateLight[x] = state.d_updateLight[x] + state.d_deltaLight[x];
				state.d_deltaLight[x] = 0.0f;
			}
		}
	}
}

extern "C" void Initialization(InputEnhanceData& state, InputEnhanceParams& params, const RayCastParams& rayCastParams)
{
	const unsigned int N = params.nImagePixel;
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	//const unsigned int padLightPixel = (N / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK;
	//const unsigned int totalLight = padLightPixel * params.nLightCoefficient * params.nLightCoefficient;
	//const int gridsPerBlockLight = (totalLight + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


	dim3 gridsPerBlockLight((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, params.nLightCoefficient, params.nLightCoefficient);

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));
	cutilSafeCall(cudaMemset(state.d_deltaLight, 0, sizeof(float) * params.nLightCoefficient));


	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, params, rayCastParams);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif	

	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> > (N, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

extern "C" float PCGEvalResidual(InputEnhanceData& state, InputEnhanceParams& params, const RayCastParams& rayCastParams)
{
	float residual_data_shading = 0.0f;
	float residual_reg = 0.0f;
	float residual_smooth = 0.0f;
	float residual_albedo_temp = 0.0f;
	float residual_albedo_spatial = 0.0f;
	const unsigned int N = params.nImagePixel;
	cudaMemset(state.d_sumResidual, 0, sizeof(float) * 8);
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (state, params, rayCastParams);

	cudaMemcpy(&residual_data_shading, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_reg, &state.d_sumResidual[3], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_smooth, &state.d_sumResidual[4], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_albedo_temp, &state.d_sumResidual[6], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_albedo_spatial, &state.d_sumResidual[7], sizeof(float), cudaMemcpyDeviceToHost);

	return residual_data_shading + residual_reg + residual_smooth + residual_albedo_spatial;
}

extern "C" float PCGEvalInitialResidual(InputEnhanceData& state, InputEnhanceParams& params, const RayCastParams& rayCastParams)
{
	float residual_data_shading = 0.0;
	float residual_reg = 0.0f;
	float residual_smooth = 0.0f;
	float residual_albedo_temp = 0.0f;
	float residual_albedo_spatial = 0.0f;
	const unsigned int N = params.nImagePixel;
	cudaMemset(state.d_sumResidual, 0, sizeof(float) * 8);
	EvalInitialResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (state, params, rayCastParams);

	cudaMemcpy(&residual_data_shading, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_reg, &state.d_sumResidual[3], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_smooth, &state.d_sumResidual[4], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_albedo_temp, &state.d_sumResidual[6], sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&residual_albedo_spatial, &state.d_sumResidual[7], sizeof(float), cudaMemcpyDeviceToHost);

	return residual_data_shading + residual_reg + residual_smooth + residual_albedo_temp + residual_albedo_spatial;
}

extern "C" void PCGEvaluation(InputEnhanceData& state, InputEnhanceParams& params, const RayCastParams& rayCastParams, int nIter = 0, int linIter = 0, int nLevel = 0) {
	// Reset the residual.
	cudaMemset(state.d_debugResidualData, 0, sizeof(float) * params.nImagePixel);
	float residual = 0;
	if (nIter == 0 && linIter == 0) {
		residual = PCGEvalInitialResidual(state, params, rayCastParams);
	}
	else {
		residual = PCGEvalResidual(state, params, rayCastParams);
	}
}


extern "C" bool PCGIteration(InputEnhanceData& state, InputEnhanceParams& enhanceParams, const RayCastParams& rayCastParams, bool lastIteration)
{
	const unsigned int N = enhanceParams.nImagePixel;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	//if (blocksPerGrid > THREADS_PER_BLOCK)
	//{
	//	std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK * THREADS_PER_BLOCK << std::endl;
	//	while (1);
	//}
	//if (timer) timer->startEvent("PCGIteration");

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	PCGStep_Kernel0 << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams, rayCastParams);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	PCGStep_Kernel1a << < blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams, rayCastParams);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel1b << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

#ifdef ENABLE_EARLY_OUT //for convergence
	float scanAlpha; cutilSafeCall(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	//if (fabs(scanAlpha) < 0.00005f) lastIteration = true;  //todo check this part
	//if (fabs(scanAlpha) < 1e-6) lastIteration = true;  //todo check this part
	if (fabs(scanAlpha) < 5e-7) { lastIteration = true; }  //todo check this part
#endif
	if (lastIteration) {
		PCGStep_Kernel3<true> << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams);
	}
	else {
		PCGStep_Kernel3<false> << <blocksPerGrid, THREADS_PER_BLOCK >> > (state, enhanceParams);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	return lastIteration;
}

extern "C" void solveEnhancementStub(InputEnhanceData& m_data, InputEnhanceParams& m_params, const RayCastParams& rayCastParams)
{

	for (unsigned int nIter = 0; nIter < m_params.nPCGOuterIteration; nIter++)
	{
		if (m_params.optimizeFull) {
			m_params.optimizeAlbedo = true;
			m_params.optimizeDepth = true;
			m_params.optimizeLight = false;
		}

		Initialization(m_data, m_params, rayCastParams);

		for (unsigned int linIter = 0; linIter < m_params.nPCGInnerIteration; linIter++)
		{
			PCGIteration(m_data, m_params, rayCastParams, linIter == (m_params.nPCGInnerIteration - 1));
#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
#endif
		}
	}
}

__global__ void UpsamplingImage(float *d_input, int inputWidth, int inputHeight, float *d_output, int outputWidth, int outputHeight, bool *d_validity, bool *d_outValidity)
{
	const unsigned int N = outputWidth * outputHeight;								// Number of block variables
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (0 <= idx && idx < N)
	{
		if (!d_outValidity[idx]) return;

		int x = idx % outputWidth;
		int y = idx / outputWidth;
		float scaleWidth = (float)(inputWidth) / (float)(outputWidth);
		float scaleHeight = (float)(inputHeight) / (float)(outputHeight);
		float scaleX = x * scaleWidth;
		float scaleY = y * scaleHeight;

		int2 p00 = make_int2(floor(scaleX), floor(scaleY));
		int2 p10 = p00 + make_int2(1, 0);
		int2 p01 = p00 + make_int2(0, 1);
		int2 p11 = p00 + make_int2(1, 1);

		float alpha = scaleX - p00.x;
		float beta = scaleY - p00.y;

		float s0 = 0.0f;
		float w0 = 0.0f;
		if (p00.x < inputWidth && p00.y < inputHeight && d_validity[p00.y * inputWidth + p00.x]) {
			float v00 = d_input[p00.y * inputWidth + p00.x];
			s0 += (1.0f - alpha) * v00;
			w0 += (1.0f - alpha);
		}
		if (p10.x < inputWidth && p10.y < inputHeight && d_validity[p10.y * inputWidth + p10.x]) {
			float v10 = d_input[p10.y * inputWidth + p10.x];
			s0 += alpha * v10;
			w0 += alpha;
		}

		float s1 = 0.0f;
		float w1 = 0.0f;
		if (p01.x < inputWidth && p01.y < inputHeight&& d_validity[p01.y * inputWidth + p01.x]) {
			float v01 = d_input[p01.y * inputWidth + p01.x];
			s1 += (1.0f - alpha) * v01;
			w1 += (1.0f - alpha);
		}
		if (p11.x < inputWidth && p11.y < inputHeight&& d_validity[p11.y * inputWidth + p11.x]) {
			float v11 = d_input[p11.y * inputWidth + p11.x];
			s1 += alpha * v11;
			w1 += alpha;
		}

		float ss = 0.0f;
		float ww = 0.0f;
		if (w0 > 0.0f) {
			float p0 = s0 / w0;
			ss += (1.0f - beta) * p0;
			ww += (1.0f - beta);
		}
		if (w1 > 0.0f) {
			float p1 = s1 / w1;
			ss += beta * p1;
			ww += beta;
		}

		if (ww > 0.0f)	d_output[y * outputWidth + x] = ss / ww;
	}
}

__global__ void UpsamplingDepthImage(float *d_input, int inputWidth, int inputHeight, float *d_srcInput, float *d_output, int outputWidth, int outputHeight, bool *d_validity, bool *d_outValidity)
{
	const unsigned int N = outputWidth * outputHeight;								// Number of block variables
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (0 <= idx && idx < N)
	{
		d_output[idx] = d_srcInput[idx];

		if (!d_outValidity[idx]) return;

		int x = idx % outputWidth;
		int y = idx / outputWidth;
		float scaleWidth = (float)(inputWidth) / (float)(outputWidth);
		float scaleHeight = (float)(inputHeight) / (float)(outputHeight);
		float scaleX = x * scaleWidth;
		float scaleY = y * scaleHeight;

		int2 p00 = make_int2(floor(scaleX), floor(scaleY));
		int2 p10 = p00 + make_int2(1, 0);
		int2 p01 = p00 + make_int2(0, 1);
		int2 p11 = p00 + make_int2(1, 1);

		float alpha = scaleX - p00.x;
		float beta = scaleY - p00.y;

		float s0 = 0.0f;
		float w0 = 0.0f;
		if (p00.x < inputWidth && p00.y < inputHeight && d_validity[p00.y * inputWidth + p00.x]) {
			float v00 = d_input[p00.y * inputWidth + p00.x];
			s0 += (1.0f - alpha) * v00;
			w0 += (1.0f - alpha);
		}
		if (p10.x < inputWidth && p10.y < inputHeight && d_validity[p10.y * inputWidth + p10.x]) {
			float v10 = d_input[p10.y * inputWidth + p10.x];
			s0 += alpha * v10;
			w0 += alpha;
		}

		float s1 = 0.0f;
		float w1 = 0.0f;
		if (p01.x < inputWidth && p01.y < inputHeight&& d_validity[p01.y * inputWidth + p01.x]) {
			float v01 = d_input[p01.y * inputWidth + p01.x];
			s1 += (1.0f - alpha) * v01;
			w1 += (1.0f - alpha);
		}
		if (p11.x < inputWidth && p11.y < inputHeight&& d_validity[p11.y * inputWidth + p11.x]) {
			float v11 = d_input[p11.y * inputWidth + p11.x];
			s1 += alpha * v11;
			w1 += alpha;
		}

		float ss = 0.0f;
		float ww = 0.0f;
		if (w0 > 0.0f) {
			float p0 = s0 / w0;
			ss += (1.0f - beta) * p0;
			ww += (1.0f - beta);
		}
		if (w1 > 0.0f) {
			float p1 = s1 / w1;
			ss += beta * p1;
			ww += beta;
		}

		if (ww > 0.0f)	d_output[y * outputWidth + x] = ss / ww;
	}
}

__global__ void UpsamplingFloat3Image(float3 *d_input, int inputWidth, int inputHeight, float3 *d_srcInput, float3 *d_output, int outputWidth, int outputHeight, bool *d_validity, bool *d_outValidity)
{
	const unsigned int N = outputWidth * outputHeight;								// Number of block variables
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (0 <= idx && idx < N)
	{
		d_output[idx] = d_srcInput[idx];

		if (!d_outValidity[idx]) return;

		int x = idx % outputWidth;
		int y = idx / outputWidth;
		float scaleWidth = (float)(inputWidth) / (float)(outputWidth);
		float scaleHeight = (float)(inputHeight) / (float)(outputHeight);
		float scaleX = x * scaleWidth;
		float scaleY = y * scaleHeight;

		int2 p00 = make_int2(floor(scaleX), floor(scaleY));
		int2 p10 = p00 + make_int2(1, 0);
		int2 p01 = p00 + make_int2(0, 1);
		int2 p11 = p00 + make_int2(1, 1);

		float alpha = scaleX - p00.x;
		float beta = scaleY - p00.y;

		float3 s0 = make_float3(0.0f);
		float w0 = 0.0f;
		if (p00.x < inputWidth && p00.y < inputHeight && d_validity[p00.y * inputWidth + p00.x]) {
			float3 v00 = d_input[p00.y * inputWidth + p00.x];
			s0 += (1.0f - alpha) * v00;
			w0 += (1.0f - alpha);
		}
		if (p10.x < inputWidth && p10.y < inputHeight && d_validity[p10.y * inputWidth + p10.x]) {
			float3 v10 = d_input[p10.y * inputWidth + p10.x];
			s0 += alpha * v10;
			w0 += alpha;
		}

		float3 s1 = make_float3(0.0f);
		float w1 = 0.0f;
		if (p01.x < inputWidth && p01.y < inputHeight&& d_validity[p01.y * inputWidth + p01.x]) {
			float3 v01 = d_input[p01.y * inputWidth + p01.x];
			s1 += (1.0f - alpha) * v01;
			w1 += (1.0f - alpha);
		}
		if (p11.x < inputWidth && p11.y < inputHeight&& d_validity[p11.y * inputWidth + p11.x]) {
			float3 v11 = d_input[p11.y * inputWidth + p11.x];
			s1 += alpha * v11;
			w1 += alpha;
		}

		float3 ss = make_float3(0.0f);
		float ww = 0.0f;
		if (w0 > 0.0f) {
			float3 p0 = s0 / w0;
			ss += (1.0f - beta) * p0;
			ww += (1.0f - beta);
		}
		if (w1 > 0.0f) {
			float3 p1 = s1 / w1;
			ss += beta * p1;
			ww += beta;
		}

		if (ww > 0.0f)	d_output[y * outputWidth + x] = ss / ww;
	}
}

__global__ void ConvertFloat3ToFloat(float3 *d_input, float *d_output, int imageWidth, int imageHeight, int N) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 <= idx && idx < N) {
		d_output[idx] = (d_input[idx].x + d_input[idx].y + d_input[idx].z) / 3.0f;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Valid Depth Normal Map Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int getPixel1D(int2 pixelIdx2D, InputEnhanceData state, InputEnhanceParams params) {
	int pixelIdx = -1;

	if (pixelIdx2D.x < 0 || pixelIdx2D.y < 0 || pixelIdx2D.x >= params.imageWidth || pixelIdx2D.y >= params.imageHeight)
		return -1;

	pixelIdx = pixelIdx2D.y * params.imageWidth + pixelIdx2D.x;

	if (state.d_srcDepth[pixelIdx] == MINF || state.d_srcDepth[pixelIdx] == 0.0f || state.d_srcImage[pixelIdx].w == MINF)
		return -1;

	return pixelIdx;
}

__device__ bool ComputeNormalValidity(int idx, InputEnhanceData state, InputEnhanceParams enhanceParams) {
	int2 p002D = getPixel2DCoordinate(idx, state, enhanceParams);
	int p00 = idx;
	int p0n = getPixel1D(p002D + make_int2(0, -1), state, enhanceParams);
	int pn0 = getPixel1D(p002D + make_int2(-1, 0), state, enhanceParams);
	int p0p = getPixel1D(p002D + make_int2(0, 1), state, enhanceParams);
	int pp0 = getPixel1D(p002D + make_int2(1, 0), state, enhanceParams);

	if (p0n < 0 || pn0 < 0 || p00 < 0 || p0p < 0 || pp0 < 0) {
		return false;
	}

	float x = p002D.x;
	float y = p002D.y;

	float fx = enhanceParams.fx;
	float fy = enhanceParams.fy;
	float cx = enhanceParams.mx;
	float cy = enhanceParams.my;

#ifdef CENTER_NORMAL
	float d00 = state.d_srcDepth[p00];
	float dn0 = state.d_srcDepth[pn0];
	float d0n = state.d_srcDepth[p0n];
	float dp0 = state.d_srcDepth[pp0];
	float d0p = state.d_srcDepth[p0p];
	float nx = (d0n + d0p) * (dp0 - dn0) / fy;
	float ny = (dn0 + dp0) * (d0p - d0n) / fx;
	float nz = ((nx * (cx - x) / fx) + (ny * (cy - y) / fy) - ((d0n + d0p) * (dn0 + dp0) / fx / fy));
#else
	float nx = state.d_srcDepth[p0n] * (state.d_srcDepth[p00] - state.d_srcDepth[pn0]) / fy;
	float ny = state.d_srcDepth[pn0] * (state.d_srcDepth[p00] - state.d_srcDepth[p0n]) / fx;
	float nz = (nx * (cx - x) / fx + ny * (cy - y) / fy - state.d_srcDepth[pn0] * state.d_srcDepth[p0n] / fx / fy);
#endif

	float3 n = make_float3(nx, ny, nz);
	float norm = length(n);

	if (norm < NORM_FLOAT_EPSILON || norm > 1.5f) {
		return false;
	}

	return true;
}

__global__ void computeDepthNormalMaskDevice(InputEnhanceData m_data, InputEnhanceParams m_params)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= 0 && x < m_params.imageWidth && y >= 0 && y < m_params.imageHeight) {
		m_data.d_srcMask[y * m_params.imageWidth + x] = false;
		bool srcMask = true;

		int2 p002D = make_int2(x, y);
		int p00 = getPixel1D(p002D, m_data, m_params);
		int p0n = getPixel1D(p002D + make_int2(0, -1), m_data, m_params);
		int pn0 = getPixel1D(p002D + make_int2(-1, 0), m_data, m_params);
		int pp0 = getPixel1D(p002D + make_int2(1, 0), m_data, m_params);
		int p0p = getPixel1D(p002D + make_int2(0, 1), m_data, m_params);
		int ppn = getPixel1D(p002D + make_int2(1, -1), m_data, m_params);
		int pnp = getPixel1D(p002D + make_int2(-1, 1), m_data, m_params);
		int ppp = getPixel1D(p002D + make_int2(1, 1), m_data, m_params);
		int pnn = getPixel1D(p002D + make_int2(-1, -1), m_data, m_params);


		if (p00 < 0 || p0n < 0 || pn0 < 0 || pp0 < 0 || p0p < 0 || ppn < 0 || pnp < 0 || ppp < 0 || pnn < 0) {
			srcMask = false;
		}
		
		if (srcMask) {
			if (abs(m_data.d_srcDepth[p00] - m_data.d_srcDepth[p0n]) > 0.1f) srcMask = false;
			if (abs(m_data.d_srcDepth[p00] - m_data.d_srcDepth[pn0]) > 0.1f) srcMask = false;
			if (abs(m_data.d_srcDepth[p00] - m_data.d_srcDepth[p0p]) > 0.1f) srcMask = false;
			if (abs(m_data.d_srcDepth[p00] - m_data.d_srcDepth[pp0]) > 0.1f) srcMask = false;
			if (!ComputeNormalValidity(p00, m_data, m_params)) srcMask = false;
		}

		if (p00 >= 0) m_data.d_srcMask[p00] = srcMask;
	}

}

extern "C" void computeDepthNormalMask(InputEnhanceData m_data, InputEnhanceParams m_params) {
	const dim3 gridSize((m_params.imageWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (m_params.imageHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeDepthNormalMaskDevice << <gridSize, blockSize >> > (m_data, m_params);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void computeOptimizationMaskDevice(InputEnhanceData m_data, InputEnhanceParams m_params)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= 0 && x < m_params.imageWidth && y >= 0 && y < m_params.imageHeight) {
		m_data.d_optimizeMask[y * m_params.imageWidth + x] = false;
		bool mask = true;

		int2 p002D = make_int2(x, y);
		int p00 = getPixel1DCoordinate(p002D, m_data, m_params);
		int p0n = getPixel1DCoordinate(p002D + make_int2(0, -1), m_data, m_params);
		int pn0 = getPixel1DCoordinate(p002D + make_int2(-1, 0), m_data, m_params);
		int pp0 = getPixel1DCoordinate(p002D + make_int2(1, 0), m_data, m_params);
		int p0p = getPixel1DCoordinate(p002D + make_int2(0, 1), m_data, m_params);
		int ppn = getPixel1DCoordinate(p002D + make_int2(1, -1), m_data, m_params);
		int pnp = getPixel1DCoordinate(p002D + make_int2(-1, 1), m_data, m_params);
		int ppp = getPixel1DCoordinate(p002D + make_int2(1, 1), m_data, m_params);
		int pnn = getPixel1DCoordinate(p002D + make_int2(-1, -1), m_data, m_params);

		if (p00 < 0 || p0n < 0 || pn0 < 0 || pp0 < 0 || p0p < 0 ||ppn < 0 || pnp < 0 || ppp < 0 || pnn < 0) {
			mask = false;
		}

		if (p00 >= 0) m_data.d_optimizeMask[p00] = mask;
	}
}

extern "C" void computeOptimizationMask(InputEnhanceData m_data, InputEnhanceParams m_params) {
	const dim3 gridSize((m_params.imageWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (m_params.imageHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeOptimizationMaskDevice << <gridSize, blockSize >> > (m_data, m_params);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void computeAlbedoTempValidMaskDevice(InputEnhanceData m_data, InputEnhanceParams m_params)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= 0 && x < m_params.imageWidth && y >= 0 && y < m_params.imageHeight) {
		int idx = y * m_params.imageWidth + x;
		m_data.d_albedoTempMask[idx] = false;

		if (m_data.d_srcMask[idx]) {
			if (m_data.d_prevAlbedo[idx].x == MINF) m_data.d_prevAlbedo[idx] = make_float4(m_params.initialAlbedo, m_params.initialAlbedo, m_params.initialAlbedo, 1.0f);
			else m_data.d_albedoTempMask[idx] = true;
		}
	}
}

extern "C" void computeAlbedoTempValidMask(InputEnhanceData m_data, InputEnhanceParams m_params) {
	const dim3 gridSize((m_params.imageWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (m_params.imageHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeAlbedoTempValidMaskDevice << <gridSize, blockSize >> > (m_data, m_params);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}