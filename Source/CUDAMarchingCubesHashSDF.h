#pragma once

#include "GlobalAppState.h"
#include "VoxelUtilHashSDF.h"
#include "MarchingCubesSDFUtil.h"
#include "CUDASceneRepChunkGrid.h"

class CUDAMarchingCubesHashSDF
{
public:
	CUDAMarchingCubesHashSDF(const MarchingCubesParams& params) {
		create(params);
	}

	~CUDAMarchingCubesHashSDF(void) {
		destroy();
	}

	static MarchingCubesParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		MarchingCubesParams params;
		params.m_maxNumTriangles = gas.s_marchingCubesMaxNumTriangles;
		params.m_threshMarchingCubes = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_threshMarchingCubes2 = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_sdfBlockSize = SDF_BLOCK_SIZE;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		return params;
	}
	
	
	void clearMeshBuffer(void) {
		m_meshData.clear();
		m_meshOnlyData.clear();
	}

	//! copies the intermediate result of extract isoSurfaceCUDA to the CPU and merges it with meshData
	//void copyTrianglesToCPU(const TexPoolData& texPoolData, const TexPoolParams& texPoolParams);
	void saveMesh(const std::string& filename, const mat4f *transform = NULL, bool overwriteExistingFile = false);
	void saveMeshWithoutTexture(const std::string & filename, const mat4f * transform = NULL, bool overwriteExistingFile = false);


	void extractIsoSurface(CUDASceneRepChunkGrid & chunkGrid, const RayCastData & rayCastData, const TexPoolData & texPoolData, const TexPoolParams & texPoolParams, const vec3f & camPos, float radius);
	void extractIsoSurface(const HashData & hashData, const HashParams & hashParams, const RayCastData & rayCastData, const TexPoolData & texPoolData, const TexPoolParams & texPoolParams, float * d_lightData, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	void extractIsoSurface(const HashData & hashData, const HashParams & hashParams, const RayCastData & rayCastData, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	
	


private:
	
	void create(const MarchingCubesParams& params);
	void destroy(void);

	void copyNoTextureTrianglesToCPU();
	void copyTrianglesToCPU(TexPoolData texPoolData, TexPoolParams texPoolParams, float * d_lightData);

	MarchingCubesParams m_params;
	MarchingCubesData	m_data;

	MeshDataf m_meshData;
	MeshDataf m_meshOnlyData;

	Timer m_timer;
};

