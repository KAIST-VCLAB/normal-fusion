#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

__align__(16)	//has to be aligned to 16 bytes
struct InputEnhanceParams {
	float fx;
	float fy;
	float mx;
	float my;

	unsigned int imageWidth;
	unsigned int imageHeight;
	unsigned int nImagePixel;

	float originalFx;
	float originalFy;
	float originalMx;
	float originalMy;

	unsigned int originalImageWidth;
	unsigned int originalImageHeight;
	unsigned int originalNumImagePixel;

	unsigned int nLightCoefficient;
	unsigned int currentLevel;
	unsigned int levelParams;

	bool optimizeAlbedo;
	bool optimizeLight;
	bool optimizeDepth;
	bool optimizeFull;

	float weightDataShading;
	float weightDepthTemp;
	float weightDepthSmooth;
	float weightAlbedoTemp;
	float weightAlbedoSmooth;
	float weightLightTemp;

	float ambientLightBase;

	int nPCGInnerIteration;
	int nPCGOuterIteration;
	int nLevel;

	bool isFirstIter;

	float initialAlbedo;
};
