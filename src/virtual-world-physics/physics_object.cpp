#include "physics_object.h"
#include <glm/gtc/matrix_transform.hpp>

namespace physics {
glm::mat4 PhysicsObject::GetModelMatrix() const {
  glm::mat4 modelMatrix = glm::mat4(1.0f);
  modelMatrix = glm::translate(modelMatrix, position);
  modelMatrix = glm::scale(modelMatrix, scale);
  return modelMatrix;
}

void PhysicsObject::UpdateBoundingVolume() {
	// TODO

	// update the position for the center
	boundingVolume.center = position;
}

void PhysicsObject::ApplyForce(const glm::vec3 &force) {
	// TODO
	if (isStatic)
		return;

	// add new acceleration to the current one
	acceleration += force / mass;
}

void PhysicsObject::ApplyImpulse(const glm::vec3 &impulse) {
	// TODO
	if (isStatic)
		return;

	velocity += impulse / mass;
}

void PhysicsObject::Integrate(float deltaTime) {
	// TODO

	// modify the speed
	velocity += (acceleration * deltaTime);

	// calculate the new position
	position += (velocity * deltaTime);

	// reset the acceleration
	acceleration = glm::vec3(0.0f);
}
} // namespace physics
