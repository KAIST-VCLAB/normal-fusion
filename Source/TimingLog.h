#pragma once

#include "GlobalAppState.h"
#include <iostream>

#define BENCHMARK_SAMPLES 128

class TimingLog
{	
	public:

		static void init()
		{
			resetTimings();
		}

		static void destroy()
		{
		}

		static void printTimings()
		{
			if (GlobalAppState::get().s_timingsDetailledEnabledOurs)
			{
				if (countTimeICP != 0)				std::cout << "Total Time ICP: " << totalTimeICP / countTimeICP << std::endl;
				if (countTimeEnhanceInput != 0)		std::cout << "Total Time Enhance Input: " << totalTimeEnhanceInput / countTimeEnhanceInput << std::endl;
				if (countTimeIntegrateDepth != 0)	std::cout << "Total Time Integrate Depth: " << totalTimeIntegrateDepth / countTimeIntegrateDepth << std::endl;
				if (countTimeTransferTexture != 0)	std::cout << "Total Time Transfer Texture: " << totalTimeTransferTexture / countTimeTransferTexture << std::endl;
				if (countTimeOptimizeTexture != 0)	std::cout << "Total Time Optimize Texture: " << totalTimeOptimizeTexture / countTimeOptimizeTexture << std::endl;
				if (countTimeBlendingTexture != 0)	std::cout << "Total Time Blending Texture: " << totalTimeBlendingTexture / countTimeBlendingTexture << std::endl;
				std::cout << std::endl; std::cout << std::endl;
			}

			if(GlobalAppState::get().s_timingsDetailledEnabled)
			{	
				if(countTimeHoleFilling != 0)		std::cout << "Total Time Hole Filling: "		<< totalTimeHoleFilling/countTimeHoleFilling			<< std::endl;
				if(countTimeFilterColor != 0)		std::cout << "Total Time Filter Color: "		<< totalTimeFilterColor/countTimeFilterColor			<< std::endl;
				if(countTimeFilterDepth != 0)		std::cout << "Total Time Filter Depth: "		<< totalTimeFilterDepth/countTimeFilterDepth			<< std::endl;
				//if(countTimeOptimizer != 0)			std::cout << "Total Time Optimizer: "			<< totalTimeOptimizer/countTimeOptimizer				<< std::endl;
				if(countTimeRGBDAdapter != 0)		std::cout << "Total Time RGBD Adapter: "		<< totalTimeRGBDAdapter/countTimeRGBDAdapter			<< std::endl;
				if(countTimeICP != 0)				std::cout << "Total Time ICP: "					<< totalTimeICP/countTimeICP							<< std::endl;
				if(countTimeEnhanceInput != 0)		std::cout << "Total Time Enhance Input: "		<< totalTimeEnhanceInput/countTimeEnhanceInput			<< std::endl;
				if(countTimeIntegrateDepth != 0)	std::cout << "Total Time Integrate Depth: "		<< totalTimeIntegrateDepth/countTimeIntegrateDepth		<< std::endl;
				if(countTimeTransferTexture != 0)	std::cout << "Total Time Transfer Texture: "	<< totalTimeTransferTexture/countTimeTransferTexture	<< std::endl;
				if(countTimeOptimizeTexture != 0)	std::cout << "Total Time Optimize Texture: "	<< totalTimeOptimizeTexture/countTimeOptimizeTexture	<< std::endl;
				if(countTimeBlendingTexture != 0)	std::cout << "Total Time Blending Texture: "	<< totalTimeBlendingTexture/countTimeBlendingTexture	<< std::endl;

				//if(countTimeClusterColor != 0)		std::cout << "Total Time Cluster Color: "		<< totalTimeClusterColor/countTimeClusterColor			<< std::endl;
				//if(countTimeEstimateLighting != 0)	std::cout << "Total Time Estimate Lighting: "	<< totalTimeEstimateLighting/countTimeEstimateLighting	<< std::endl;
				//if(countTimeRemapDepth != 0)		std::cout << "Total Time Remap Depth: "			<< totalTimeRemapDepth/countTimeRemapDepth				<< std::endl;
				//if(countTimeSegment != 0)			std::cout << "Total Time Segment Depth: "		<< totalTimeSegment/countTimeSegment					<< std::endl;
				//if(countTimeRender != 0)			std::cout << "Total Time Render: "	 			<< totalTimeRender/countTimeRender						<< std::endl;
				if(countTimeRayIntervalSplatting != 0) std::cout << "Total Time RayIntervalSplatting: "	<< totalTimeRayIntervalSplatting/countTimeRayIntervalSplatting << std::endl;
				//if(countTimeRayIntervalSplattingCUDA != 0) std::cout << "Total Time RayIntervalSplatting (CUDA): "	<< totalTimeRayIntervalSplattingCUDA/countTimeRayIntervalSplattingCUDA << std::endl;
				//if(countTimeRayIntervalSplattingDX11 != 0) std::cout << "Total Time RayIntervalSplatting (DX11): "	<< totalTimeRayIntervalSplattingDX11/countTimeRayIntervalSplattingDX11 << std::endl;
				if(countTimeRayCast != 0)			std::cout << "Total Time RayCast: "	 			<< totalTimeRayCast/countTimeRayCast					<< std::endl;
				if(countTimeTracking != 0)			std::cout << "Total Time Tracking: "			<< totalTimeTracking/countTimeTracking					<< std::endl;
				if(countTimeSFS != 0)				std::cout << "Total Time SFS: "					<< totalTimeSFS/countTimeSFS							<< std::endl;
				if(countTimeCompactifyHash != 0)	std::cout << "Total Time Compactify Hash: "		<< totalTimeCompactifyHash/countTimeCompactifyHash		<< std::endl;
				if(countTimeAlloc != 0)				std::cout << "Total Time Alloc: "				<< totalTimeAlloc/countTimeAlloc						<< std::endl;
				if(countTimeIntegrate != 0)			std::cout << "Total Time Integrate: "			<< totalTimeIntegrate/countTimeIntegrate				<< std::endl;

				std::cout << std::endl; std::cout << std::endl;
			}

			if(GlobalAppState::get().s_timingsTotalEnabled)
			{
				if(countTotalTimeAll != 0)
				{
					double avg = 0.0;
					for(unsigned int i = 0; i < std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll); i++)
					{
						avg += totalTimeAllAvgArray[i];
					}

					avg /= std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll);
					
					if(countTotalTimeAll >= BENCHMARK_SAMPLES)
					{
						totalTimeAllMaxAvg = std::max(totalTimeAllMaxAvg, avg);
						totalTimeAllMinAvg = std::min(totalTimeAllMinAvg, avg);
					}

					std::cout << "count: " << countTotalTimeAll << std::endl;
					std::cout << "Time Window Avg:\t" << avg << std::endl;
					std::cout << "Time All Avg Total:\t" << totalTimeAll/countTotalTimeAll << std::endl;
					double deviation = sqrt((totalTimeSquaredAll/countTotalTimeAll)-(totalTimeAll/countTotalTimeAll)*(totalTimeAll/countTotalTimeAll));
					std::cout << "Time All Std.Dev.:\t" << deviation << std::endl;

					std::cout << "Time All Worst:\t" << totalTimeAllWorst << std::endl;

					std::cout << std::endl;
				}
			}
		}

		static void printTimingOneFrame()
		{
			if (GlobalAppState::get().s_timingsDetailledEnabled)
			{
				if (countTimeHoleFilling != 0)		std::cout << "Total Time Hole Filling: " << totalTimeHoleFilling << std::endl;
				if (countTimeFilterColor != 0)		std::cout << "Total Time Filter Color: " << totalTimeFilterColor << std::endl;
				if (countTimeFilterDepth != 0)		std::cout << "Total Time Filter Depth: " << totalTimeFilterDepth << std::endl;
				//if(countTimeOptimizer != 0)			std::cout << "Total Time Optimizer: "			<< totalTimeOptimizer/countTimeOptimizer				<< std::endl;
				if (countTimeRGBDAdapter != 0)		std::cout << "Total Time RGBD Adapter: " << totalTimeRGBDAdapter << std::endl;
				if (countTimeEnhanceInput != 0)		std::cout << "Total Time Enhance Input: " << totalTimeEnhanceInput << std::endl;
				//if(countTimeClusterColor != 0)		std::cout << "Total Time Cluster Color: "		<< totalTimeClusterColor/countTimeClusterColor			<< std::endl;
				//if(countTimeEstimateLighting != 0)	std::cout << "Total Time Estimate Lighting: "	<< totalTimeEstimateLighting/countTimeEstimateLighting	<< std::endl;
				//if(countTimeRemapDepth != 0)		std::cout << "Total Time Remap Depth: "			<< totalTimeRemapDepth/countTimeRemapDepth				<< std::endl;
				//if(countTimeSegment != 0)			std::cout << "Total Time Segment Depth: "		<< totalTimeSegment/countTimeSegment					<< std::endl;
				//if(countTimeRender != 0)			std::cout << "Total Time Render: "	 			<< totalTimeRender/countTimeRender						<< std::endl;
				if (countTimeRayIntervalSplatting != 0) std::cout << "Total Time RayIntervalSplatting: " << totalTimeRayIntervalSplatting << std::endl;
				//if(countTimeRayIntervalSplattingCUDA != 0) std::cout << "Total Time RayIntervalSplatting (CUDA): "	<< totalTimeRayIntervalSplattingCUDA/countTimeRayIntervalSplattingCUDA << std::endl;
				//if(countTimeRayIntervalSplattingDX11 != 0) std::cout << "Total Time RayIntervalSplatting (DX11): "	<< totalTimeRayIntervalSplattingDX11/countTimeRayIntervalSplattingDX11 << std::endl;
				if (countTimeRayCast != 0)			std::cout << "Total Time RayCast: " << totalTimeRayCast << std::endl;
				if (countTimeTracking != 0)			std::cout << "Total Time Tracking: " << totalTimeTracking << std::endl;
				if (countTimeSFS != 0)				std::cout << "Total Time SFS: " << totalTimeSFS << std::endl;
				if (countTimeCompactifyHash != 0)	std::cout << "Total Time Compactify Hash: " << totalTimeCompactifyHash << std::endl;
				if (countTimeAlloc != 0)				std::cout << "Total Time Alloc: " << totalTimeAlloc << std::endl;
				if (countTimeIntegrate != 0)			std::cout << "Total Time Integrate: " << totalTimeIntegrate << std::endl;

				std::cout << std::endl; std::cout << std::endl;
			}

			if (GlobalAppState::get().s_timingsTotalEnabled)
			{
				if (countTotalTimeAll != 0)
				{
					double avg = 0.0;
					for (unsigned int i = 0; i < std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll); i++)
					{
						avg += totalTimeAllAvgArray[i];
					}

					avg /= std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll);

					if (countTotalTimeAll >= BENCHMARK_SAMPLES)
					{
						totalTimeAllMaxAvg = std::max(totalTimeAllMaxAvg, avg);
						totalTimeAllMinAvg = std::min(totalTimeAllMinAvg, avg);
					}

					std::cout << "count: " << countTotalTimeAll << std::endl;
					std::cout << "Time Window Avg:\t" << avg << std::endl;
					std::cout << "Time All Avg Total:\t" << totalTimeAll / countTotalTimeAll << std::endl;
					double deviation = sqrt((totalTimeSquaredAll / countTotalTimeAll) - (totalTimeAll / countTotalTimeAll)*(totalTimeAll / countTotalTimeAll));
					std::cout << "Time All Std.Dev.:\t" << deviation << std::endl;

					std::cout << "Time All Worst:\t" << totalTimeAllWorst << std::endl;

					std::cout << std::endl;
				}
			}
		}

		static void resetTimings()
		{
			totalTimeHoleFilling = 0.0;
			countTimeHoleFilling = 0;

			totalTimeRender = 0.0;
			countTimeRender = 0;

			totalTimeOptimizer = 0.0;
			countTimeOptimizer = 0;

			totalTimeFilterColor = 0.0;
			countTimeFilterColor = 0;

			totalTimeFilterDepth = 0.0;
			countTimeFilterDepth = 0;

			totalTimeRGBDAdapter = 0.0;
			countTimeRGBDAdapter = 0;

			totalTimeClusterColor = 0.0;
			countTimeClusterColor = 0;

			totalTimeEstimateLighting = 0.0;
			countTimeEstimateLighting = 0;

			totalTimeRemapDepth = 0.0;
			countTimeRemapDepth = 0;

			totalTimeSegment = 0.0;
			countTimeSegment = 0;

			totalTimeTracking = 0.0;
			countTimeTracking = 0;

			totalTimeSFS = 0.0;
			countTimeSFS = 0;

			totalTimeRayCast = 0.0;
			countTimeRayCast = 0;

			totalTimeRayIntervalSplatting = 0.0;
			countTimeRayIntervalSplatting = 0;

			totalTimeICP = 0.0;
			countTimeICP = 0;

			totalTimeEnhanceInput = 0.0;
			countTimeEnhanceInput = 0;

			totalTimeIntegrateDepth = 0.0;
			countTimeIntegrateDepth = 0;

			totalTimeTransferTexture = 0.0;
			countTimeTransferTexture = 0;

			totalTimeOptimizeTexture = 0.0f;
			countTimeOptimizeTexture = 0;

			totalTimeBlendingTexture = 0.0f;
			countTimeBlendingTexture = 0;

			//totalTimeRayIntervalSplattingCUDA = 0.0;
			//countTimeRayIntervalSplattingCUDA = 0;

			//totalTimeRayIntervalSplattingDX11 = 0.0;
			//countTimeRayIntervalSplattingDX11 = 0;

			totalTimeCompactifyHash = 0.0;
			countTimeCompactifyHash = 0;

			totalTimeAlloc = 0.0;
			countTimeAlloc = 0;

			totalTimeIntegrate = 0.0;
			countTimeIntegrate = 0;

			for(unsigned int i = 0; i < BENCHMARK_SAMPLES; i++) totalTimeAllAvgArray[i] = 0.0;

			// Benchmark
			countTotalTimeAll = 0;
			totalTimeAllWorst = 0.0;
			totalTimeAllMaxAvg = 0.0;
			totalTimeAllMinAvg = std::numeric_limits<double>::max();
			totalTimeAll = 0.0;
			totalTimeSquaredAll = 0.0;
		}

		static double totalTimeHoleFilling;
		static unsigned int countTimeHoleFilling;

		static double totalTimeRender;
		static unsigned int countTimeRender;

		static double totalTimeOptimizer;
		static unsigned int countTimeOptimizer;

		static double totalTimeFilterColor;
		static unsigned int countTimeFilterColor;

		static double totalTimeFilterDepth;
		static unsigned int countTimeFilterDepth;

		static double totalTimeRGBDAdapter;
		static unsigned int countTimeRGBDAdapter;

		static double totalTimeClusterColor;
		static unsigned int countTimeClusterColor;

		static double totalTimeEstimateLighting;
		static unsigned int countTimeEstimateLighting;

		static double totalTimeRemapDepth;
		static unsigned int countTimeRemapDepth;

		static double totalTimeSegment;
		static unsigned int countTimeSegment;

		// camera tracking
		static double totalTimeTracking;
		static unsigned int countTimeTracking;

		static double totalTimeSFS;
		static unsigned int countTimeSFS;

		// ray cast
		static double totalTimeRayCast;
		static unsigned int countTimeRayCast;

		static double totalTimeRayIntervalSplatting;
		static unsigned int countTimeRayIntervalSplatting;

		static double totalTimeICP;
		static unsigned int countTimeICP;

		static double totalTimeEnhanceInput;
		static unsigned int countTimeEnhanceInput;

		static double totalTimeIntegrateDepth;
		static unsigned int countTimeIntegrateDepth;

		static double totalTimeTransferTexture;
		static unsigned int countTimeTransferTexture;

		static double totalTimeOptimizeTexture;
		static unsigned int countTimeOptimizeTexture;

		static double totalTimeBlendingTexture;
		static unsigned int countTimeBlendingTexture;
		//static double totalTimeRayIntervalSplattingCUDA;
		//static unsigned int countTimeRayIntervalSplattingCUDA;

		//static double totalTimeRayIntervalSplattingDX11;
		//static unsigned int countTimeRayIntervalSplattingDX11;

		// scene rep
		static double totalTimeCompactifyHash;
		static unsigned int countTimeCompactifyHash;
		
		static double totalTimeAlloc;
		static unsigned int countTimeAlloc;
		
		static double totalTimeIntegrate;
		static unsigned int countTimeIntegrate;

		//benchmark
		static double totalTimeAllAvgArray[BENCHMARK_SAMPLES];

		static unsigned int countTotalTimeAll;
		static double totalTimeAllWorst;
		static double totalTimeAllMaxAvg;
		static double totalTimeAllMinAvg;
		static double totalTimeAll;
		static double totalTimeSquaredAll;
};
