
#include <cutil_inline.h>
#include <cutil_math.h>

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "MarchingCubesSDFUtil.h"
#include "texturePool.h"


__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
	*data.d_numOccupiedBlocks = 0;	
}
 
__global__ void extractIsoSurfaceKernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data, TexPoolData texPoolData) 
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData, texPoolData);
	}
}

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);

	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, const TexPoolData& texPoolData)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data, texPoolData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}




///new

#define T_PER_BLOCK 8

//tags all
__global__ void extractIsoSurfacePass1Kernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x;


	if (bucketID < hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {

		//HashEntry entry = getHashEntry(g_Hash, bucketID);
		HashEntry& entry = hashData.d_hash[bucketID];

		if (entry.ptr != FREE_ENTRY) {

			//float3 pos = hashData.SDFBlockToWorld(entry.pos);
			//float l = SDF_BLOCK_SIZE*hashParams.m_virtualVoxelSize;
			//float3 minCorner = data.d_params->m_minCorner - l;
			//float3 maxCorner = data.d_params->m_maxCorner;
			//
			//if (data.d_params->m_boxEnabled == 1) {
			//	if (!data.isInBoxAA(minCorner, maxCorner, pos)) return;
			//}

			uint addr = atomicAdd(&data.d_numOccupiedBlocks[0], 1);
			data.d_occupiedBlocks[addr] = bucketID;
		}
	}
}

extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize((params.m_hashNumBuckets*params.m_hashBucketSize + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);


	extractIsoSurfacePass1Kernel << <gridSize, blockSize >> >(hashData, rayCastData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void extractIsoSurfacePass2Kernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data, TexPoolData texPoolData)
{
	//const HashParams& hashParams = c_hashParams;
	uint idx = data.d_occupiedBlocks[blockIdx.x];

	//if (idx >= hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {
	//	if (threadIdx.x == 0) {
	//		printf("%d: invalid idx! %d\n", blockIdx.x, idx);
	//	}
	//	return;
	//}
	//return;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData, texPoolData);
	}
}


__global__ void extractIsoSurfacePass2WithoutTextureKernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data)
{
	//const HashParams& hashParams = c_hashParams;
	uint idx = data.d_occupiedBlocks[blockIdx.x];

	//if (idx >= hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {
	//	if (threadIdx.x == 0) {
	//		printf("%d: invalid idx! %d\n", blockIdx.x, idx);
	//	}
	//	return;
	//}
	//return;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPositionWithoutTexture(worldPos, hashData, rayCastData);
	}
}

extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, const TexPoolData& texPoolData, unsigned int numOccupiedBlocks)
{
	const dim3 gridSize(numOccupiedBlocks, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	if (numOccupiedBlocks) {
		extractIsoSurfacePass2Kernel << <gridSize, blockSize >> >(hashData, rayCastData, data, texPoolData);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void extractIsoSurfacePass2WithoutTextureCUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks)
{
	const dim3 gridSize(numOccupiedBlocks, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	if (numOccupiedBlocks) {
		extractIsoSurfacePass2WithoutTextureKernel << <gridSize, blockSize >> > (hashData, rayCastData, data);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void
exportTextureDevice(uchar *d_output, uint *d_texelAddress, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches)
{
	const int patch_x = blockIdx.x;
	const int patch_y = blockIdx.y;
	const int patch_idx = patch_y * tex_width + patch_x;

	if (patch_idx >= num_patches) return;

	const int texture_texel_x = threadIdx.x;
	const int texture_texel_y = threadIdx.y;
	int texel_x = texture_texel_x - 1;
	int texel_y = texture_texel_y - 1;

	if (texel_x < 0) texel_x = 0;
	if (texel_x >= patch_width) texel_x = patch_width - 1;
	if (texel_y < 0) texel_y = 0;
	if (texel_y >= patch_width) texel_y = patch_width - 1;

	const int texture_pixel_x = patch_x * (patch_width + 2) + texture_texel_x;
	const int texture_pixel_y = patch_y * (patch_width + 2) + texture_texel_y;

	const int texture_pixel_width = tex_width * (patch_width + 2);

	const int texture_pixel_idx = texture_pixel_y * texture_pixel_width + texture_pixel_x;
	const int texel_index = d_texelAddress[patch_idx] * patch_size + texel_y * patch_width + texel_x;

	//d_output[3 * texture_pixel_idx] = d_input[texel_index].color.x;
	//d_output[3 * texture_pixel_idx + 1] = d_input[texel_index].color.y;
	//d_output[3 * texture_pixel_idx + 2] = d_input[texel_index].color.z;

	d_output[3 * texture_pixel_idx] = powf((d_input[texel_index].color.x / 255.f), 1.0f / 2.2f) * 255.f;
	d_output[3 * texture_pixel_idx + 1] = powf((d_input[texel_index].color.y / 255.f), 1.0f / 2.2f) * 255.f;
	d_output[3 * texture_pixel_idx + 2] = powf((d_input[texel_index].color.z / 255.f), 1.0f / 2.2f) * 255.f;
}

extern "C" void exportTexture(uchar *d_output, uint *d_texelAddress, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches)
{
	const dim3 gridSize(tex_width, tex_width);
	const dim3 blockSize(patch_width + 2, patch_width + 2);

	exportTextureDevice << <gridSize, blockSize >> > (d_output, d_texelAddress, d_input, tex_width, tex_height, patch_width, patch_size, num_patches, max_num_patches);
}

__device__ float computeTextureShading(float *lightCoeffs, float3 n) {
	float sum = 0.f;

	sum += lightCoeffs[0];
	sum += lightCoeffs[1] * n.y;
	sum += lightCoeffs[2] * n.z;
	sum += lightCoeffs[3] * n.x;
	sum += lightCoeffs[4] * n.x * n.y;
	sum += lightCoeffs[5] * n.y * n.z;
	sum += lightCoeffs[6] * (-n.x * n.x - n.y * n.y + 2.f * n.z * n.z);
	sum += lightCoeffs[7] * n.z * n.x;
	sum += lightCoeffs[8] * (n.x * n.x - n.y * n.y);

	return sum;
}

__global__ void
exportDoubleTextureDevice(uchar *d_output, uchar *d_outputAlbedo, uchar *d_normalOutput, uint *d_texelAddress, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches, float *d_lightData)
{
	const int patch_x = blockIdx.x;
	const int patch_y = blockIdx.y;
	const int patch_idx = patch_y * tex_width + patch_x;

	if (patch_idx >= num_patches) return;

	const int texture_texel_x = threadIdx.x;
	const int texture_texel_y = threadIdx.y;
	int texel_x = texture_texel_x - 1;
	int texel_y = texture_texel_y - 1;

	if (texel_x < 0) texel_x = 0;
	if (texel_x >= patch_width) texel_x = patch_width - 1;
	if (texel_y < 0) texel_y = 0;
	if (texel_y >= patch_width) texel_y = patch_width - 1;

	const int texture_pixel_x = patch_x * (patch_width + 2) + texture_texel_x;
	const int texture_pixel_y = patch_y * (patch_width + 2) + texture_texel_y;

	const int texture_pixel_width = tex_width * (patch_width + 2);

	const int texture_pixel_idx = texture_pixel_y * texture_pixel_width + texture_pixel_x;
	const int texel_index = d_texelAddress[patch_idx] * patch_size + texel_y * patch_width + texel_x;

	float3 normal = d_input[texel_index].normal_texture;
	
	if (normal.x != MINF && length(normal) > 0.8f && length(normal) < 1.2f) {
		float shading = computeTextureShading(d_lightData, normal); 
		d_output[3 * texture_pixel_idx] = uchar(fminf(powf((d_input[texel_index].color.x / 255.f) * shading, 1.0f / 2.2f) * 255.f, 254.9f));
		d_output[3 * texture_pixel_idx + 1] = uchar(fminf(powf((d_input[texel_index].color.y / 255.f) * shading, 1.0f / 2.2f) * 255.f, 254.9f));
		d_output[3 * texture_pixel_idx + 2] = uchar(fminf(powf((d_input[texel_index].color.z / 255.f) * shading, 1.0f / 2.2f) * 255.f, 254.9f));
		d_outputAlbedo[3 * texture_pixel_idx] = uchar(fminf(powf((d_input[texel_index].color.x / 255.f), 1.0f / 2.2f) * 255.f, 254.9f));
		d_outputAlbedo[3 * texture_pixel_idx + 1] = uchar(fminf(powf((d_input[texel_index].color.y / 255.f), 1.0f / 2.2f) * 255.f, 254.9f));
		d_outputAlbedo[3 * texture_pixel_idx + 2] = uchar(fminf(powf((d_input[texel_index].color.z / 255.f), 1.0f / 2.2f) * 255.f, 254.9f));
		d_normalOutput[3 * texture_pixel_idx] = (-d_input[texel_index].normal_texture.z + 1.0f) * 0.5f * 255.f;
		d_normalOutput[3 * texture_pixel_idx + 1] = (-d_input[texel_index].normal_texture.y + 1.0f) * 0.5f * 255.f;
		d_normalOutput[3 * texture_pixel_idx + 2] = (d_input[texel_index].normal_texture.x + 1.0f) * 0.5f * 255.f;
	}
	else {
		d_output[3 * texture_pixel_idx] = 0.f;
		d_output[3 * texture_pixel_idx + 1] = 0.f;
		d_output[3 * texture_pixel_idx + 2] = 0.f;
		d_outputAlbedo[3 * texture_pixel_idx] = 0.f;
		d_outputAlbedo[3 * texture_pixel_idx + 1] = 0.f;
		d_outputAlbedo[3 * texture_pixel_idx + 2] = 0.f;
		d_normalOutput[3 * texture_pixel_idx] = 0.f;
		d_normalOutput[3 * texture_pixel_idx + 1] = 0.f;
		d_normalOutput[3 * texture_pixel_idx + 2] = 0.f;
	}
}

extern "C" void exportDoubleTexture(uchar *d_output, uchar *d_outputAlbedo, uchar *d_normalOutput, uint *d_texelAddress, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches, float *d_lightData)
{
	const dim3 gridSize(tex_width, tex_width);
	const dim3 blockSize(patch_width + 2, patch_width + 2);

	exportDoubleTextureDevice << <gridSize, blockSize >> > (d_output, d_outputAlbedo, d_normalOutput, d_texelAddress, d_input, tex_width, tex_height, patch_width, patch_size, num_patches, max_num_patches, d_lightData);
}