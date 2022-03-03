# NormalFusion: Real-Time Acquisition of Surface Normals for High-Resolution RGB-D Scanning

#### [Project Page](http://vclab.kaist.ac.kr/cvpr2021p3/index.html) | [Paper](http://vclab.kaist.ac.kr/cvpr2021p3/cvpr2021-normalfusion-main-paper.pdf) | [Supplemental material #1](http://vclab.kaist.ac.kr/cvpr2021p3/cvpr2021-normalfusion-supple.pdf) | [Supplemental material #2](https://youtu.be/7l6oVDK5Dz4) | [Presentation Video](https://youtu.be/_JCtwGOriyg)

Hyunho Ha (hhha@vclab.kaist.ac.kr), Joo Ho Lee (jhlee@vclab.kaist.ac.kr), Andreas Meuleman (ameuleman@vclab.kaist.ac.kr) and Min H. Kim (minhkim@kaist.ac.kr)

Institute: KAIST Visual Computing Laboratory

If you use our code for your academic work, please cite our paper:

	@InProceedings{Ha_2021_CVPR,
		author = {Hyunho Ha and Joo Ho Lee and Andreas Meuleman and Min H. Kim},
		title = {NormalFusion: Real-Time Acquisition of Surface Normals for High-Resolution RGB-D Scanning},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2021}
	}

## Installation

Our implementation is based on the original voxel hashing (https://github.com/niessner/VoxelHashing) and TextureFusion code (https://github.com/KAIST-VCLAB/texturefusion).

To compile our codes, first obtain the entire source codes from the original [voxel hashing repository](https://github.com/niessner/VoxelHashing), including the Visual Studio project file. Then follow these steps:

1. In `VoxelHashing/DepthSensingCUDA/`, replace the folders `Include/`, `Source/` (excluding `Source/NewFileList/`), and `Shaders/` as well as the configuration files `zParameters*.txt` with the contents of our repository.

2. Replace `DepthSensing.cpp` and `DepthSensing.h` file with `normalFusion.h`, `normalFusion.cpp`, and `normalFusion_main.cpp`

3. Configure the existing files in the `Source/*.h`, `Source/*.cpp`, and `Source/*.cu` to the Visual Studio project that does not exist in the voxel hashing repository. A list of the newly added codes is duplicated in `Source/NewFileList/`.

Note that our source codes inherit the dependency of the original Voxel Hashing project.

Our work requires:
- [DirectX SDK June 2010](https://www.microsoft.com/en-us/download/details.aspx?id=6812)
- Both [Kinect SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) and [Kinect SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (tested with version 10.1)
- Both [mLib](https://github.com/niessner/mLib) and mLibExternal (http://kaldir.vc.in.tum.de/mLib/mLibExternal.zip) with [OpenCV](https://opencv.org/) (tested with version 3.4.1): Note that the zip file, mLibExternal, includes other dependent libraries such as OpenNI 2 and Eigen.

Our code has been developed with Microsoft Visual Studio 2013 (VC++ 12) and Windows 10 (10.0.19041, build 19041) on a machine equipped with Intel i9-10920X (RAM: 64GB), NVIDIA TITAN RTX (RAM: 24GB). The main function is in `normalFusion_main.cpp`.\

## Data

We provide the "fountain" dataset (originally created by [Zhou and Koltun](http://qianyi.info/scenedata.html)) compatible with our implementation
(link: [http://vclab.kaist.ac.kr/cvpr2020p1/fountain_all.zip](http://vclab.kaist.ac.kr/cvpr2020p1/fountain_all.zip)).

## Usage

Our program reads parameters from three files and you can change the program setting by changing them.

- zParametersDefault.txt

- zParametersTrackingDefault.txt

- zParametersWarpingDefault.txt

- zParametersEnhancementDefault.txt

You can run our program with the provided fountain dataset.

Please set s_sensorIdx as 9 and s_binaryDumpSensorFile[0] as the fountain folder in zParametersDefault.txt.

Our program produces mesh with two textures (diffuse albedo and normal). If you want to further enhance mesh using normal texture, please refer to the paper: 
"Efficiently Combining Positions and Normals for Precise 3D Geometry", Nehab et al., ACM TOG, 2005.

## License

Hyunho Ha, Joo Ho Lee, Andreas Meuleman, and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.

The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].

Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

Note that Our implementation inherits the original license of "Voxel Hashing" codes (CC BY-NC-SA 3.0). 

Please refer to license.txt for more details. 

## Contact

If you have any questions, please feel free to contact us.

Hyunho Ha (hhha@vclab.kaist.ac.kr)

Joo Ho Lee (jhlee@vclab.kaist.ac.kr)

Andreas Meuleman (ameuleman@vclab.kaist.ac.kr)

Min H. Kim (minhkim@vclab.kaist.ac.kr)
