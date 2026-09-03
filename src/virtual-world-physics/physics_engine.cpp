#include "physics_engine.h"
#include "virtual-world-physics/bounding_volume.h"
#include "virtual-world-physics/collision.h"
#include "virtual-world-physics/physics_gpu.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace physics {
void PhysicsEngine::ClearObjects() { m_objects.clear(); }

void PhysicsEngine::Update(float deltaTime) {
  // Reset statistics
  m_stats.detectedCollisions = 0;
  m_stats.objectCount = static_cast<int>(m_objects.size());

  auto startTime = std::chrono::high_resolution_clock::now();

  // Fixed timestep with accumulator
  m_accumulator += deltaTime;
  int subSteps = 0;
  while (m_accumulator >= m_fixedDeltaTime && subSteps < m_maxSubSteps) {
    ApplyGravity(m_fixedDeltaTime);
    for (auto &object : m_objects) {
      object.Integrate(m_fixedDeltaTime);
      object.UpdateBoundingVolume();
    }
    if (m_useGPU && m_gpuDetector) {
      try {
        auto collisions = m_gpuDetector->DetectCollisions(m_objects);

        m_stats.detectedCollisions = collisions.size();

        for (const auto &collision : collisions) {
          ResolveCollision(collision.indexA, collision.indexB, collision);
        }
      } catch (const std::exception &ex) {
        std::cerr << "CUDA collision backend failed, falling back to CPU: "
                  << ex.what() << std::endl;
        m_useGPU = false;
        BroadPhase();
        NarrowPhase();
      }
    } else {
      BroadPhase();
      NarrowPhase();
    }

    m_accumulator -= m_fixedDeltaTime;
    subSteps++;
  }
  if (m_accumulator > m_fixedDeltaTime) {
    m_accumulator = 0.0f;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  m_stats.collisionDetectionTime =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void PhysicsEngine::ApplyGravity(float deltaTime) {
    // TODO

    // apply gravity to each object
    for (PhysicsObject& object: m_objects) {
        object.ApplyForce(object.mass * m_gravity);
    }
}

void PhysicsEngine::BroadPhase() {  
    // TODO

    // the position modified after Integrate function
    for (PhysicsObject& object : m_objects) {
        object.UpdateBoundingVolume();
    }

    collisionPairs = PhysicsEngine::GetPotentialCollisionPairs();
}


std::vector<std::pair<size_t, size_t>>
PhysicsEngine::GetPotentialCollisionPairs() {
    // TODO

    std::vector<size_t> indices;

    for (int i = 0; i < m_objects.size(); i++)
        indices.push_back(i);

    // sort by X
    std::sort(indices.begin(), indices.end(), [this](size_t first, size_t second) {
        const auto& boxA = m_objects[first].boundingVolume;
        const auto& boxB = m_objects[second].boundingVolume;

        return (boxA.center.x - boxA.sizes.x) < (boxB.center.x - boxB.sizes.x);
    });

    std::vector<std::pair<size_t, size_t>> pairs;

    for (int i = 0; i < indices.size() - 1; i++) {
        for (int j = i + 1; j < indices.size(); j++) {
            const auto& boxA = m_objects[indices[i]].boundingVolume;
            const auto& boxB = m_objects[indices[j]].boundingVolume;
            
            // maxA comparison with minB on X axis
            if (boxA.center.x + boxA.sizes.x < boxB.center.x - boxB.sizes.x)
                break;

            // verify if they intersect
            if (Intersects(boxA, boxB)) {
                pairs.push_back({ indices[i], indices[j] });
            }
        }
    }

    return pairs;
}

void PhysicsEngine::NarrowPhase() {
    // TODO

    // verify each pair
    for (const auto& collisionPair: collisionPairs) {
        CollisionInfo info = DetectCollision(collisionPair.first, collisionPair.second);

        if (glm::length(info.normal) > 0.0f) {
            m_stats.detectedCollisions++;
            ResolveCollision(collisionPair.first, collisionPair.second, info);
        }
    }
}

CollisionInfo PhysicsEngine::DetectCollision(size_t indexA, size_t indexB) {
    // TODO
    
    const auto& boxA = m_objects[indexA].boundingVolume;
    const auto& boxB = m_objects[indexB].boundingVolume;

    return ComputeBoxBoxCollision(indexA, indexB, boxA, boxB);
}

CollisionInfo
PhysicsEngine::ComputeBoxBoxCollision(size_t indexA, size_t indexB,
                                      const bounding_volume_t &boxA,
                                      const bounding_volume_t &boxB) {
    // TODO

    CollisionInfo info;
    info.indexA = indexA;
    info.indexB = indexB;

    glm::vec3 distance = boxB.center - boxA.center;
    glm::vec3 absDistance = glm::abs(distance);
    glm::vec3 sizes = boxA.sizes + boxB.sizes;

    glm::vec3 intersection = absDistance - sizes;

    // they intersect
    if (intersection.x <= 0 && intersection.y <= 0 && intersection.z <= 0) {
        // the shortest path to separate the two object
        if (intersection.x >= intersection.y && intersection.x >= intersection.z) {
            info.normal = glm::vec3(distance.x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
            info.penetration = -intersection.x;
        }
        else if (intersection.y >= intersection.z) {
            info.normal = glm::vec3(0.0f, distance.y > 0 ? 1.0f : -1.0f, 0.0f);
            info.penetration = -intersection.y;
        }
        else {
            info.normal = glm::vec3(0.0f, 0.0f, distance.z > 0 ? 1.0f : -1.0f);
            info.penetration = -intersection.z;
        }
    }

    return info;
}

void PhysicsEngine::ResolveCollision(size_t indexA, size_t indexB,
                                     const CollisionInfo &collision) {
    // TODO

    auto& objectA = m_objects[indexA];
    auto& objectB = m_objects[indexB];

    if (objectA.isStatic && objectB.isStatic)
        return;

    float totalMass = objectA.mass + objectB.mass;

    float ratioA = 0.0f;
    float ratioB = 0.0f;
   
    // depends on each mass
    if (objectA.isStatic && !objectB.isStatic) {
        ratioA = 0.0f;
        ratioB = 1.0f;
    }
    else if (!objectA.isStatic && objectB.isStatic) {
        ratioA = 1.0f;
        ratioB = 0.0f;
    }
    else if (!objectA.isStatic && !objectB.isStatic) {
        float totalMass = objectA.mass + objectB.mass;
        ratioA = objectB.mass / totalMass;
        ratioB = objectA.mass / totalMass;
    }

    glm::vec3 correction = collision.penetration * collision.normal;

    if (!objectA.isStatic) objectA.position -= correction * ratioA;
    if (!objectB.isStatic) objectB.position += correction * ratioB;

    // application of impulse via relative velocity
    glm::vec3 relVelocity = objectB.velocity - objectA.velocity;
    float velocityNormal = glm::dot(relVelocity, collision.normal);

    if (velocityNormal > 0)
        return;

    float effectiveMass;
    
    if (objectA.isStatic)
        effectiveMass = objectB.mass;
    else if (objectB.isStatic)
        effectiveMass = objectA.mass;
    else
        effectiveMass = (objectA.mass * objectB.mass) / totalMass;

    // normal impulse / min elasticity
    float e = std::min(objectA.restitution, objectB.restitution);
    if (std::abs(velocityNormal) < 0.2f) {
        e = 0.0f;
    }

    // magnitude of the frontal impact
    float mag = -(1.0f + e) * velocityNormal * effectiveMass;
    glm::vec3 normalImpulse = mag * collision.normal;

    if (!objectA.isStatic) objectA.ApplyImpulse(-normalImpulse);
    if (!objectB.isStatic) objectB.ApplyImpulse(normalImpulse);

    glm::vec3 tangent = relVelocity - (velocityNormal * collision.normal);

    if (glm::length(tangent) > 0.000001f) {
        tangent = glm::normalize(tangent);

        float magt = -glm::dot(relVelocity, tangent) * effectiveMass;
        float mu = (objectA.friction + objectB.friction) / 2.0f;

        glm::vec3 frictionImpulse;

        if (std::abs(magt) < mag * mu)
            frictionImpulse = magt * tangent;
        else
            frictionImpulse = -mag * mu * tangent;

        if (!objectA.isStatic) objectA.ApplyImpulse(-frictionImpulse);
        if (!objectB.isStatic) objectB.ApplyImpulse(frictionImpulse);
    }
}

void PhysicsEngine::Init(const glm::vec3 &gravity, bool useGpu) {
  m_gravity = gravity;
  m_useGPU = false;

  delete m_gpuDetector;
  m_gpuDetector = nullptr;

  if (useGpu) {
    m_gpuDetector = new GPUCollisionDetector();
    if (m_gpuDetector->Initialize()) {
      m_useGPU = true;
    } else {
      delete m_gpuDetector;
      m_gpuDetector = nullptr;
      std::cerr << "CUDA collision backend unavailable; using CPU backend."
                << std::endl;
    }
  }
}

PhysicsEngine::~PhysicsEngine() { delete m_gpuDetector; }
} // namespace physics
