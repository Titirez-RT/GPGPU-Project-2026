#pragma once
#include "physics_gpu.h"
#include <cuda_runtime.h>
#include "virtual-world-physics/bounding_volume.h"


// Define here functions and auxiliaries for GPU physics computations.

namespace physics {
	void StartCollision(const bounding_volume_t* volumes, int objectsNum,
		CollisionInfo* result, int* count, int maxCollisions);
}
