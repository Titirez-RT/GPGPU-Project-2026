#include "physics_gpu.h"
#include "physics_gpu.cuh"

// Define here kernels and entry points for GPU physics computations.

namespace physics {
	__global__ void Collisions(const bounding_volume_t* volumes, int objectsNum,
		CollisionInfo* result, int* count, int maxCollisions) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i >= objectsNum || j >= objectsNum || i >= j)
			return;

		bounding_volume_t boxA = volumes[i];
		bounding_volume_t boxB = volumes[j];

		float x = boxB.center.x - boxA.center.x;
		float y = boxB.center.y - boxA.center.y;
		float z = boxB.center.z - boxA.center.z;

		float intersectionX = fabs(x) - (boxA.sizes.x + boxB.sizes.x);
		float intersectionY = fabs(y) - (boxA.sizes.y + boxB.sizes.y);
		float intersectionZ = fabs(z) - (boxA.sizes.z + boxB.sizes.z);

		// they intersect
		if (intersectionX <= 0 && intersectionY <= 0 && intersectionZ <= 0) {
			int idx = atomicAdd(count, 1);

			if (idx < maxCollisions) {
				CollisionInfo info;
				info.isValid = true;
				info.indexA = i;
				info.indexB = j;

				// the shortest path to separate the two object
				if (intersectionX >= intersectionY && intersectionX >= intersectionZ) {
					info.normal = glm::vec3(x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
					info.penetration = -intersectionX;
				}
				else if (intersectionY >= intersectionZ) {
					info.normal = glm::vec3(0.0f, y > 0 ? 1.0f : -1.0f, 0.0f);
					info.penetration = -intersectionY;
				}
				else {
					info.normal = glm::vec3(0.0f, 0.0f, z > 0 ? 1.0f : -1.0f);
					info.penetration = -intersectionZ;
				}

				result[idx] = info;
			}
		}
	}


	void StartCollision(const bounding_volume_t* volumes, int objectsNum,
		CollisionInfo* result, int* count, int maxCollisions) {

		dim3 blockSize(16, 16);
		dim3 gridSize((objectsNum + blockSize.x - 1) / blockSize.x,
			(objectsNum + blockSize.y - 1) / blockSize.y);

		Collisions << <gridSize, blockSize >> > (volumes, objectsNum, result, count, maxCollisions);
	}

	bool GPUCollisionDetector::Initialize() {
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		m_initialized = deviceCount > 0;
		return m_initialized;
	}

	GPUCollisionDetector::~GPUCollisionDetector() {
		if (m_currentCapacity > 0) {
			cudaFree(deviceVolumes);
			cudaFree(deviceCollisionResult);
			cudaFree(destCounter);

			deviceVolumes = nullptr;
			deviceCollisionResult = nullptr;
			destCounter = nullptr;
			m_currentCapacity = 0;
		}
	}

	std::vector<CollisionInfo> GPUCollisionDetector::DetectCollisions(const std::vector<PhysicsObject>& objects) {
		if (!m_initialized || objects.empty())
			return {};

		int objectsNum = objects.size();
		int maxCollisions = objectsNum * 100;

		if (objectsNum > m_currentCapacity) {
			if (m_currentCapacity > 0) {
				cudaFree(deviceVolumes);
				cudaFree(deviceCollisionResult);
				cudaFree(destCounter);
			}

			cudaMalloc(&deviceVolumes, objectsNum * sizeof(bounding_volume_t));
			cudaMalloc(&deviceCollisionResult, maxCollisions * sizeof(CollisionInfo));
			cudaMalloc(&destCounter, sizeof(int));
			m_currentCapacity = objectsNum;
		}


		std::vector<bounding_volume_t> hostVolumes(objectsNum);
		for (int i = 0; i < objectsNum; i++) {
			hostVolumes[i] = objects[i].boundingVolume;
		}

		// transfer data Host -> Device
		cudaMemcpy(deviceVolumes, hostVolumes.data(), objectsNum * sizeof(bounding_volume_t), cudaMemcpyHostToDevice);
		cudaMemset(destCounter, 0, sizeof(int));

		StartCollision(deviceVolumes, objectsNum, deviceCollisionResult, destCounter, maxCollisions);

		// transfer data Device -> Host
		int realCollisions = 0;
		cudaMemcpy(&realCollisions, destCounter, sizeof(int), cudaMemcpyDeviceToHost);

		realCollisions = std::min(realCollisions, maxCollisions);
		std::vector<CollisionInfo> results(realCollisions);
		if (realCollisions > 0)
			cudaMemcpy(results.data(), deviceCollisionResult, realCollisions * sizeof(CollisionInfo), cudaMemcpyDeviceToHost);

		return results;
	}
}