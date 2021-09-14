#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"
#include "VoxelUtilHashSDF.h"
#include "texturePool.h"

#include "CUDARayCastParams.h"
#include "CUDATexUpdateParams.h"
#include "TexUpdateUtil.h"
#ifndef __CUDACC__
#include "mLib.h"
#endif

struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
};

#ifndef MINF
#define MINF asfloat(0xff800000)
#endif

extern __constant__ RayCastParams c_rayCastParams;
//extern __constant__ TexUpdateParams c_texUpdateParams;
extern "C" void updateConstantRayCastParams(const RayCastParams& params);


struct RayCastData {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
		RayCastData() {
		d_depth = NULL;
		d_depth4 = NULL;
		d_depthPrev4 = NULL;
		d_normals = NULL;
		d_normalsWorld = NULL;
		d_detailNormals = NULL;
		d_detailNormalsWorld = NULL;
		d_detailNormalsPrev = NULL;
		d_colors = NULL;
		d_colorVoxels = NULL;
		d_rhoD = NULL;
		d_rhoDPrev = NULL;
		d_validMask = NULL;
		d_vertexBuffer = NULL;
		d_weightMap = NULL;

		d_normalShading = NULL;
		d_detailNormalShading = NULL;

		d_rayIntervalSplatMinArray = NULL;
		d_rayIntervalSplatMaxArray = NULL;

		d_normalModelArray = NULL;
		d_rhoDModelArray = NULL;
	}

#ifndef __CUDACC__
	__host__
		void allocate(const RayCastParams& params) {
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth4, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthPrev4, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normals, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_detailNormals, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normalsWorld, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_detailNormalsWorld, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_detailNormalsPrev, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colors, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorVoxels, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rhoD, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rhoDPrev, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_validMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_weightMap, sizeof(float) * params.m_width * params.m_height));


		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normalShading, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_detailNormalShading, sizeof(float4) * params.m_width * params.m_height));

		h_normalModelChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		MLIB_CUDA_SAFE_CALL(cudaMallocArray(&d_normalModelArray, &h_normalModelChannelDesc, params.m_width, params.m_height));
		h_rhoDModelChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		MLIB_CUDA_SAFE_CALL(cudaMallocArray(&d_rhoDModelArray, &h_rhoDModelChannelDesc, params.m_width, params.m_height));
	}

	__host__
		void updateParams(const RayCastParams& params) {
		updateConstantRayCastParams(params);
	}

	__host__
		void free() {
		MLIB_CUDA_SAFE_FREE(d_depth);
		MLIB_CUDA_SAFE_FREE(d_depth4);
		MLIB_CUDA_SAFE_FREE(d_depthPrev4);
		MLIB_CUDA_SAFE_FREE(d_normals);
		MLIB_CUDA_SAFE_FREE(d_detailNormals);
		MLIB_CUDA_SAFE_FREE(d_normalsWorld);
		MLIB_CUDA_SAFE_FREE(d_detailNormalsWorld);
		MLIB_CUDA_SAFE_FREE(d_detailNormalsPrev);
		MLIB_CUDA_SAFE_FREE(d_colors);
		MLIB_CUDA_SAFE_FREE(d_colorVoxels);
		MLIB_CUDA_SAFE_FREE(d_rhoD);
		MLIB_CUDA_SAFE_FREE(d_rhoDPrev);
		MLIB_CUDA_SAFE_FREE(d_validMask);
		MLIB_CUDA_SAFE_FREE(d_weightMap);

		MLIB_CUDA_SAFE_FREE(d_normalShading);
		MLIB_CUDA_SAFE_FREE(d_detailNormalShading);


		cudaFreeArray(d_rhoDModelArray);
		cudaFreeArray(d_normalModelArray);
	}
#endif

	/////////////////
	// Device part //
	/////////////////
//#define __CUDACC__
#ifdef __CUDACC__

	__device__
		const RayCastParams& params() const {
		return c_rayCastParams;
	}

	__device__
		float frac(float val) const {
		return (val - floorf(val));
	}

	__device__
		float3 frac(const float3& val) const {
		return make_float3(frac(val.x), frac(val.y), frac(val.z));
	}

	__device__
		bool trilinearInterpolationSimpleFastFast(const HashData& hash, const float3& pos, float& dist, uchar3& color) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
		Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

		return true;

	}

	__device__
		bool trilinearInterpolationPrevSimpleFastFast(const HashData& hash, const float3& pos, float& dist, uchar3& color) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
		Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.prev_sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.prev_sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.prev_sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.prev_sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.prev_sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.prev_sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

		return true;

	}

	__device__
		bool trilinearInterpolationPrevSimpleFastFast2(const HashData& hash, const float3& pos, float& dist) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f); // relative position of position
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos)); // a - floor (a)

		dist = 0.0f;
		Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;     dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   dist += weight.x *	   weight.y *(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   dist += weight.x *(1.0f - weight.y)*	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   dist += weight.x *	   weight.y *	   weight.z *v.prev_sdf;

		return true;

	}

	__device__
		float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const
	{
		return tNear + (dNear / (dNear - dFar))*(tFar - tNear);
	}

	static const unsigned int nIterationsBisection = 3;

	// d0 near, d1 far
	__device__
		bool findIntersectionBisection(const HashData& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for (uint i = 0; i < nIterationsBisection; i++)
		{
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;
			if (!trilinearInterpolationSimpleFastFast(hash, worldCamPos + c*worldDir, cDist, color)) return false;

			if (aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}

	// d0 near, d1 far
	__device__
		bool findIntersectionBisectionPrev(const HashData& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for (uint i = 0; i < nIterationsBisection; i++)
		{
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;
			if (!trilinearInterpolationPrevSimpleFastFast(hash, worldCamPos + c * worldDir, cDist, color)) return false;

			if (aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}
	__device__
		float3 gradientForPoint(const HashData& hash, const float3& pos) const
	{
		const float voxelSize = c_hashParams.m_virtualVoxelSize;
		float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

		float distp00; uchar3 colorp00; trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
		float dist0p0; uchar3 color0p0; trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
		float dist00p; uchar3 color00p; trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

		float dist100; uchar3 color100; trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100, color100);
		float dist010; uchar3 color010; trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010, color010);
		float dist001; uchar3 color001; trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001, color001);

		float3 grad = make_float3((distp00 - dist100) / offset.x, (dist0p0 - dist010) / offset.y, (dist00p - dist001) / offset.z);

		float l = length(grad);
		if (l == 0.0f) {
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		return -grad / l;
	}

	__device__
		float3 gradientForPointPrev(const HashData& hash, const float3& pos) const
	{
		const float voxelSize = c_hashParams.m_virtualVoxelSize*0.5;
		float3 offset = make_float3(voxelSize, voxelSize, voxelSize);
		bool valid = true;
		float distp00; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos - make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00);
		float dist0p0; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos - make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0);
		float dist00p; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos - make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p);

		float dist100; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos + make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100);
		float dist010; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos + make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010);
		float dist001; valid &= trilinearInterpolationPrevSimpleFastFast2(hash, pos + make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001);

		float3 grad = make_float3((distp00 - dist100) / offset.x, (dist0p0 - dist010) / offset.y, (dist00p - dist001) / offset.z);

		float l = length(grad);
		if (l == 0.0f || !valid) {
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		// - two times!!!!
		return -grad / l;
	}
	__device__
		void traverseCoarseGridSimpleSampleAll(const HashData& hash, const DepthCameraData& cameraData, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length

		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
		//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
		//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength

#pragma unroll 1
		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos + rayCurrent*worldDir;
			float dist;	uchar3 color;

			if (trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					uchar3 color2;
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);

					float3 currentIso = worldCamPos + alpha*worldDir;
					if (b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
					{
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width + dTid.x] = depth;
							d_depth4[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(cameraData.kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);
							d_colors[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(color2.x / 255.f, color2.y / 255.f, color2.z / 255.f, 1.0f);

							if (rayCastParams.m_useGradients)
							{
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				// lastSample.color = color;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}


		}

	}

	/////////////////////////////////////////////////////////////////////////
	//Texture transfer & texture rendering
	/////////////////////////////////////////////////////////////////////////

	__device__
		bool trilinearInterpolationSimpleFastFastPrev(const HashData& hash, const float3& pos, float& dist) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f); // relative position of position
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos)); // a - floor (a)

		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
		Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.prev_weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.prev_sdf;
		v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.prev_weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.prev_sdf;

		return true;
	}

	__device__
		bool findIntersectionBisectionPrev(const HashData& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for (uint i = 0; i < nIterationsBisection; i++)
		{

			//find zero point by line assumption.
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;

			if (!trilinearInterpolationSimpleFastFastPrev(hash, worldCamPos + c*worldDir, cDist)) return false;

			if (aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}

	__device__
		void traverseCoarseGridSimpleSampleAllFromTexture(const HashData& hash, const TexPoolData& texPoolData, const DepthCameraData& cameraData, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length

		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
																							//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
																							//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength
		d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;

#pragma unroll 1
		
		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos + rayCurrent*worldDir;
			float dist;	uchar3 color;

			if (trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))//color: voxel color
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{
					//dist : next interpolated sdf value
					//lastSample: last interpolated sdf value
					//last alpha, new alpha

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color);
					
					float3 currentIso = worldCamPos + alpha*worldDir;

					if (b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist) //there is bisection and the step is under threshold.
					{
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width + dTid.x] = depth;
							d_depth4[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(cameraData.kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);

							//project current iso to the texture space and get texel information
							Texel texel = texPoolData.getTexel(currentIso, hash);
							
							//if (texel.alpha < -1.f) { //if a texel is not valid
							//if (texel.color.x == MINF ) { //if a texel is not valid
							if (texel.color.x <= 0.01) { //if a texel is not valid
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;
								d_colors[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);
							}
							else { // for valid texel
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 0.f;
								d_colors[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(texel.color.x / 255.f, texel.color.y / 255.f, texel.color.z / 255.f, 1.0f);
							}

							//project the weight texture to the camera plane.
							d_weightMap[dTid.y*rayCastParams.m_width + dTid.x] = texel.weight;

							if (rayCastParams.m_useGradients)
							{

								//why they put - here? 
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);

							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
		}
	}

	__device__ float3x3 RodriguesRotation(float3 v1, float3 v2) const {
		float3x3 identity;
		identity.setZero();
		identity(0, 0) = 1.0f; identity(1, 1) = 1.0f; identity(2, 2) = 1.0f;
		//return identity;

		float3 v = cross(v1, v2);
		float3x3 crossMat;
		crossMat.setZero();
		crossMat(0, 1) = -v.z; crossMat(0, 2) = v.y;
		crossMat(1, 0) = v.z; crossMat(1, 2) = -v.x;
		crossMat(2, 0) = -v.y; crossMat(2, 1) = v.x;

		float c = dot(v1, v2);
		float s = length(v);

		float3x3 rotationMatrix = identity + crossMat + crossMat * crossMat * ((1.0f - c) / (s * s));
		return rotationMatrix;
	}

	//used
	__device__
		void traverseCoarseGridSimpleSamplePrevTexture(const HashData& hash, const TexPoolData& texPoolData, const DepthCameraData& cameraData, const float3& worldPos, const float3& worldDir, const int& texel_addr, float minInterval, float maxInterval, const float3x3& rotMat) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;
		const TexUpdateParams& texUpdateParams = c_texUpdateParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);

		float rayCurrent = minInterval;	// Convert depth to raylength
		float rayEnd = maxInterval;		// Convert depth to raylength
										//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
										//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength


#pragma unroll 1
		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldPos + rayCurrent*worldDir;
			float dist;


			if (trilinearInterpolationSimpleFastFastPrev(hash, currentPosWorld, dist))
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{
					//dist : next interpolated sdf value
					//lastSample: last interpolated sdf value
					//last alpha, new alpha

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					bool b = findIntersectionBisectionPrev(hash, worldPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha);
					float3 currentIso = worldPos + alpha*worldDir;
					if (b && abs(lastSample.sdf - dist) <rayCastParams.m_thresDist) //there is bisection and the step is under threshold.
					{
						//					printf("*");
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							//Voxel v = hash.getVoxel(worldPos);
							Texel texelPrev = texPoolData.getTexelPrev(currentIso, hash);
							
							texPoolData.d_texPatches[texel_addr].color = texelPrev.color;
							texPoolData.d_texPatches[texel_addr].color_dummy = texelPrev.color_dummy;
							texPoolData.d_texPatches[texel_addr].weight = texelPrev.weight;

							if (texPoolData.d_texPatches[texel_addr].weight > 0 && texelPrev.normal_texture.x != MINF) {
								texPoolData.d_texPatches[texel_addr].normal_texture = normalize(rotMat * texelPrev.normal_texture);
							}
								//if (texPoolData.d_texPatches[texel_addr].weight > 0 && texelPrev.normal_texture.x != MINF)	texPoolData.d_texPatches[texel_addr].normal_texture = texelPrev.normal_texture;
							else texPoolData.d_texPatches[texel_addr].normal_texture = make_float3(MINF, MINF, MINF);
						}
					}
				}
				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				// lastSample.color = color;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
		}
	}

	__device__
		float computeDiffuseShading(float *lightCoeffs, float4 n) const {

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

	__device__
		void traverseCoarseGridSimpleSampleAllFromDoubleTexture(const HashData& hash, const TexPoolData& texPoolData, const DepthCameraData& cameraData, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length

		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
																							//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
																							//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength
		d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;

#pragma unroll 1

		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
			float dist;	uchar3 color;

			if (trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))//color: voxel color
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{
					//dist : next interpolated sdf value
					//lastSample: last interpolated sdf value
					//last alpha, new alpha

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color);

					float3 currentIso = worldCamPos + alpha * worldDir;

					if (b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist) //there is bisection and the step is under threshold.
					{
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width + dTid.x] = depth;
							d_depth4[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(cameraData.kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);

							//project current iso to the texture space and get texel information
							Texel texel = texPoolData.getTexel(currentIso, hash);

							//if (texel.alpha < -1.f) { //if a texel is not valid
							//if (texel.color.x == MINF ) { //if a texel is not valid
							if (texel.alpha <= -1.f || texel.normal_texture.x == MINF) { //if a texel is not valid
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;
								d_colors[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);
								d_rhoD[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);
								d_detailNormals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);
								d_detailNormalsWorld[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);

							}
							else { // for valid texel
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 0.f;
								
								float4 n = rayCastParams.m_viewMatrix * make_float4(normalize(texel.normal_texture), 0.0f);
								d_detailNormals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(normalize(make_float3(n)), 1.0f);
								d_detailNormalsWorld[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(texel.normal_texture, 1.0f);

								d_rhoD[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(texel.color.x / 255.f, texel.color.y / 255.f, texel.color.z / 255.f, 1.0f);
							}

							//project the weight texture to the camera plane.
							d_weightMap[dTid.y*rayCastParams.m_width + dTid.x] = texel.weight;

							if (rayCastParams.m_useGradients)
							{

								//why they put - here? 
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
								if (!(texel.color.x >= 0.01))
									d_colors[dTid.y*rayCastParams.m_width + dTid.x] = d_rhoD[dTid.y*rayCastParams.m_width + dTid.x] * computeDiffuseShading(cameraData.d_lightData, make_float4(normal, 0.0f));

							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
		}
	}

	__device__
		void traverseCoarseGridSimpleSampleAllFromDoubleTextureGeometry(const HashData& hash, const TexPoolData& texPoolData, const DepthCameraData& cameraData, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length

		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
																							//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
																							//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength
		d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;

#pragma unroll 1

		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
			float dist;	uchar3 color;

			if (trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))//color: voxel color
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{
					//dist : next interpolated sdf value
					//lastSample: last interpolated sdf value
					//last alpha, new alpha

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color);

					float3 currentIso = worldCamPos + alpha * worldDir;

					if (b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist) //there is bisection and the step is under threshold.
					{
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength
							d_depth4[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(cameraData.kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);
							d_colorVoxels[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(color.x / 255.f, color.y / 255.f, color.z / 255.f, 1.0f);
							if (rayCastParams.m_useGradients)
							{

								//why they put - here? 
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
		}
	}

	__device__
		void traverseCoarseGridSimpleSampleAllFromPrevDoubleTexture(const HashData& hash, const TexPoolData& texPoolData, const DepthCameraData& cameraData, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length

		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
																							//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
																							//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength
		d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;

#pragma unroll 1

		while (rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
			float dist;	uchar3 color;

			if (trilinearInterpolationPrevSimpleFastFast(hash, currentPosWorld, dist, color))//color: voxel color
			{
				if (lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{
					//dist : next interpolated sdf value
					//lastSample: last interpolated sdf value
					//last alpha, new alpha

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					//bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color);
					bool b = findIntersectionBisectionPrev(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color);

					float3 currentIso = worldCamPos + alpha * worldDir;

					if (b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist) //there is bisection and the step is under threshold.
					{
						if (abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depthPrev4[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(cameraData.kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);

							//project current iso to the texture space and get texel information
							Texel texel = texPoolData.getTexel(currentIso, hash);

							//if (texel.alpha < -1.f) { //if a texel is not valid
							//if (texel.color.x == MINF ) { //if a texel is not valid
							if (texel.alpha <= -1.f || texel.normal_texture.x == MINF) { //if a texel is not valid
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 1.f;
								d_detailNormalsPrev[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);
								d_rhoDPrev[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(MINF, MINF, MINF, MINF);

							}
							else { // for valid texel
								d_validMask[dTid.y*rayCastParams.m_width + dTid.x] = 0.f;
								float4 n = rayCastParams.m_viewMatrix * make_float4(normalize(texel.normal_texture), 0.0f);
								d_detailNormalsPrev[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(normalize(make_float3(n)), 1.0f);
								d_rhoDPrev[dTid.y*rayCastParams.m_width + dTid.x] = make_float4(texel.color.x / 255.f, texel.color.y / 255.f, texel.color.z / 255.f, 1.0f);
							}

							//project the weight texture to the camera plane.
							d_weightMap[dTid.y*rayCastParams.m_width + dTid.x] = texel.weight;

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
			else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}
		}
	}

#endif // __CUDACC__

	float*  d_depth;
	float4* d_depth4;
	float4* d_depthPrev4;
	float4* d_normals;
	float4* d_detailNormals;
	float4* d_normalsWorld;
	float4* d_detailNormalsWorld;
	float4* d_detailNormalsPrev;
	float4* d_colors;
	float4* d_colorVoxels;
	float4* d_rhoD;
	float4* d_rhoDPrev;
	float* d_validMask;
	float* d_weightMap;


	float4* d_normalShading;
	float4* d_detailNormalShading;

	float4* d_vertexBuffer; // ray interval splatting triangles, mapped from directx (memory lives there)

	cudaArray* d_rayIntervalSplatMinArray;
	cudaArray* d_rayIntervalSplatMaxArray;

	cudaArray*	d_normalModelArray;
	cudaArray*	d_rhoDModelArray;
	cudaChannelFormatDesc h_normalModelChannelDesc;
	cudaChannelFormatDesc h_rhoDModelChannelDesc;
};
