#pragma once
#pragma once

/***********************************************************************************/
/* Global App state for camera tracking: reads and stores all tracking parameters  */
/***********************************************************************************/

#include "DXUT.h"

#include "stdafx.h"

#include <vector>

#define X_GLOBAL_ENHANCE_STATE_FIELDS \
	X(unsigned int, s_maxLevels) \
	X(std::vector<unsigned int>, s_maxOuterIter) \
	X(std::vector<unsigned int>, s_maxInnerIter) \
	X(std::vector<float>, s_weightsDataShading) \
	X(std::vector<float>, s_weightsDepthTemp) \
	X(std::vector<float>, s_weightsDepthSmooth) \
	X(std::vector<float>, s_weightsAlbedoTemp) \
	X(std::vector<float>, s_weightsAlbedoSmooth) \
	X(std::vector<float>, s_weightsLightTemp) \
	X(float, s_ambientLightBase) \
	X(bool, s_optimizeDepth) \
	X(bool, s_optimizeAlbedo) \
	X(bool, s_optimizeLight) \
	X(bool, s_optimizeFull) \
	X(float, s_initialAlbedo)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif


class GlobalEnhancementState
{
public:
#define X(type, name) type name;
	X_GLOBAL_ENHANCE_STATE_FIELDS
#undef X

		GlobalEnhancementState() {
		setDefault();
	}

	//! setting default parameters
	void setDefault() {
		s_maxOuterIter.resize(1);
		s_maxInnerIter.resize(1);

		s_weightsDataShading.resize(1);
		s_weightsDepthTemp.resize(1);
		s_weightsDepthSmooth.resize(1);
		s_weightsAlbedoTemp.resize(1);
		s_weightsAlbedoSmooth.resize(1);
		s_weightsLightTemp.resize(1);

		s_maxLevels = 1;
		s_maxOuterIter[0] = 10;
		s_maxInnerIter[0] = 5;

		s_weightsDataShading[0] = 1.0f;
		s_weightsDepthTemp[0] = 10.0f;
		s_weightsDepthSmooth[0] = 100.0f;
		s_weightsAlbedoTemp[0] = 0.001f;
		s_weightsAlbedoSmooth[0] = 0.01f;
		s_weightsLightTemp[0] = 50.0f;

		s_optimizeDepth = false;
		s_optimizeAlbedo = false;
		s_optimizeLight = false;
		s_optimizeFull = true;
		s_initialAlbedo = 0.5f;
		s_ambientLightBase = 1.0f;
	}

	//! sets the parameter file and reads
	void readMembers(const ParameterFile& parameterFile) {
		s_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_ENHANCE_STATE_FIELDS
#undef X
	}

	//! prints all members
	void print() {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_ENHANCE_STATE_FIELDS
#undef X
	}

	static GlobalEnhancementState& getInstance() {
		static GlobalEnhancementState s;
		return s;
	}

	static GlobalEnhancementState& get() {
		return getInstance();
	}
private:
	ParameterFile s_ParameterFile;
}; 
