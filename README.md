#### Implementation Details
The physics engine was implemented on both CPU and GPU following these steps:

**1. Kinematics and Basic Forces (`physics_object.cpp`)**
*   `ApplyForce`: Added the applied force to the acceleration (`F/m`).
*   `ApplyImpulse`: Modified the velocity based on the impulse.
*   `Integrate`: Updated the velocity and position, then reset the acceleration to zero.

**2. Global Forces (`physics_engine.cpp`)**
*   `ApplyGravity`: Iterated through objects and applied gravity.

**3. Bounding Volumes (AABB)**
*   `UpdateBoundingVolume` (`physics_object.cpp`): Synchronized the volume's center with the object's new position.
*   `Intersects` (`collision.h`): Implemented the AABB vs AABB intersection test on all 3 axes.

**4. Collision Detection: Broad Phase (`physics_engine.cpp`)**
*   `BroadPhase`: Updated bounding volumes and called potential collision pairs.
*   `GetPotentialCollisionPairs`: Implemented the Sweep and Prune algorithm (sorted objects on the X-axis for optimization and verified intersections).

**5. Collision Detection: Narrow Phase (`physics_engine.cpp`)**
*   `ComputeBoxBoxCollision`: Calculated the exact minimum penetration axis and collision normal.
*   `DetectCollision`: Extracted object references and got intersection info.
*   `NarrowPhase`: Iterated through Broad Phase pairs and sent the intersecting ones for resolution.

**6. Collision Resolution (`physics_engine.cpp`)**
*   `ResolveCollision`:
    *   **Positional correction:** Separated objects based on mass ratio to prevent sinking.
    *   **Normal impulse:** Calculated relative velocity along the normal and applied the restitution impulse.
    *   **Friction impulse:** Calculated and applied friction along the tangent of impact.

**7. GPU Parallelization - CUDA (`physics_gpu_cuda.cu`)**
*   `DetectCollisions` (Host): Allocated memory on the device (`cudaMalloc`), transferred bounding volume data from Host to Device, launched the collision kernel and transferred results (collision count and array) back to Host.
*   `Collisions` (GPU Kernel): Mathematically computed intersections and normals in parallel on grids/blocks, using `atomicAdd` for concurrent writing of valid `CollisionInfo` to the results array.

**Performance Note:** 
By pressing the 'G' key during the simulation, you can toggle between the CPU and GPU implementations. The performance difference is highly noticeable in the framerate, demonstrating the massive speedup achieved through GPU parallelization.