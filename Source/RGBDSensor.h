#pragma once

#include <windows.h>
#include <cassert>
#include "Eigen.h"
#include <d3dx9math.h>

#include "mLib.h"


namespace ml {
	class SensorData;
	class RGBDFrameCacheWrite;
}

class RGBDSensor
{
public:

	// Default constructor
	RGBDSensor();

	//! Init
	void init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight, unsigned int depthRingBufferSize = 1);

	//! Destructor; releases allocated ressources
	virtual ~RGBDSensor();

	//! Connected to Depth Sensor
	virtual HRESULT createFirstConnected() = 0;

	//! Processes the depth data 
	virtual HRESULT processDepth() = 0;

	//! Processes the color data
	virtual HRESULT processColor() = 0;

	//! Returns the sensor name
	virtual std::string getSensorName() const = 0;

	//! Toggles the near-mode if available
	virtual HRESULT toggleNearMode();

	//! Get the intrinsic camera matrix of the depth sensor
	const mat4f& getDepthIntrinsics() const;
	const mat4f& getDepthIntrinsicsInv() const;

	//! Get the intrinsic camera matrix of the depth sensor
	const mat4f& getColorIntrinsics() const;
	const mat4f& getColorIntrinsicsInv() const;
	const mat4f& getDepthExtrinsics() const;
	const mat4f& getDepthExtrinsicsInv() const;
	const mat4f& getColorExtrinsics() const;
	const mat4f& getColorExtrinsicsInv() const;

	void initializeDepthIntrinsics(float fovX, float fovY, float centerX, float centerY);
	void initializeColorIntrinsics(float fovX, float fovY, float centerX, float centerY);

	void initializeDepthExtrinsics(const Matrix3f& R, const Vector3f& t);
	void initializeColorExtrinsics(const Matrix3f& R, const Vector3f& t);
	void initializeDepthExtrinsics(const mat4f& m);
	void initializeColorExtrinsics(const mat4f& m);

	void incrementRingbufIdx();

	//! gets the pointer to depth array
	float*			getDepthFloat();
	const float*	getDepthFloat() const;

	//! gets the pointer to color array
	vec4uc*			getColorRGBX();
	const vec4uc*	getColorRGBX() const;

	unsigned int getColorWidth() const;
	unsigned int getColorHeight() const;
	unsigned int getDepthWidth() const;
	unsigned int getDepthHeight() const;

	virtual void reset(); 

	//! saves the point cloud of the current frame to a file
	void savePointCloud(const std::string& filename, const mat4f& transform = mat4f::identity()) const;

	//! records the current frame to an internally
	void recordFrame();

	//! records current transform
	void recordTrajectory(const mat4f& transform);

	//! accumulates
	void recordPointCloud(const mat4f& transform = mat4f::identity());
	void saveRecordedPointCloud(const std::string& filename);

	//! saves all previously recorded frames to file
	void saveRecordedFramesToFile(const std::string& filename);

	//! returns the current rigid transform; if not specified by the 'actual' sensor the identiy is returned
	virtual mat4f getRigidTransform(int offset) const {
		return mat4f::identity();
	}

	virtual void startReceivingFrames() {}
	virtual void stopReceivingFrames() {}
protected:

	unsigned int m_currentRingBufIdx;

	mat4f m_depthIntrinsics;
	mat4f m_depthIntrinsicsInv;

	mat4f m_depthExtrinsics;
	mat4f m_depthExtrinsicsInv;

	mat4f m_colorIntrinsics;
	mat4f m_colorIntrinsicsInv;

	mat4f m_colorExtrinsics;
	mat4f m_colorExtrinsicsInv;

	std::vector<float*> m_depthFloat;
	vec4uc*				m_colorRGBX;

	LONG   m_depthWidth;
	LONG   m_depthHeight;

	LONG   m_colorWidth;
	LONG   m_colorHeight;

	bool   m_bNearMode;


private:
	void computePointCurrentPointCloud(PointCloudf& pc, const mat4f& transform = mat4f::identity()) const;
	vec3f depthToSkeleton(unsigned int ux, unsigned int uy) const;
	vec3f depthToSkeleton(unsigned int ux, unsigned int uy, float depth) const;
	vec3f getNormal(unsigned int x, unsigned int y) const;


	bool m_bUseModernSensFilesForRecording;

	std::list<float*> m_recordedDepthData;
	std::list<vec4uc*>	m_recordedColorData;

	std::vector<mat4f> m_recordedTrajectory;
	std::list<PointCloudf> m_recordedPoints;

	//new recording version
	ml::SensorData* m_recordedData;
	ml::RGBDFrameCacheWrite* m_recordedDataCache;
};
