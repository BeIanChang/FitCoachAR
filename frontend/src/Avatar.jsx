import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// This component will render our 3D model
function Model(props) {
  // useGLTF is a hook from drei that loads GLTF models
  const { scene, nodes, animations } = useGLTF('/avatar.glb');
  
  // We'll need to access the bones of the skeleton to animate them
  const skeleton = nodes.Hips.parent;

    // This hook runs on every frame, allowing us to update the model in real-time
  useFrame(() => {
    if (props.landmarks && props.landmarks.length > 0) {
      const landmarks = props.landmarks;

      // Helper function to get a landmark's position as a THREE.Vector3
      const getLandmark = (index) => {
        const lm = landmarks[index];
        // MediaPipe landmarks are normalized; we'll scale them for our scene
        // We also flip the Y coordinate because WebGL's Y is up, and MediaPipe's is down.
        return new THREE.Vector3(lm.x * 2 - 1, -lm.y * 2 + 1, lm.z);
      };

      // --- Bone Rotation Logic ---
      const updateBoneRotation = (startIdx, midIdx, endIdx, boneName) => {
        const startPoint = getLandmark(startIdx);
        const midPoint = getLandmark(midIdx);
        const endPoint = getLandmark(endIdx);

        const bone = nodes[boneName];
        if (!bone) return;

        // Calculate the vectors for the parent and child segments
        const parentVector = midPoint.clone().sub(startPoint);
        const childVector = endPoint.clone().sub(midPoint);

        // Calculate the rotation required to align the bone
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0), // Assuming the bone's default orientation is pointing up
          childVector.normalize()
        );

        // Apply the rotation
        bone.quaternion.slerp(quaternion, 0.1); // slerp for smooth animation
      };
      
      // --- Animate the whole body ---
      // Note: The bone names ('LeftArm', 'RightUpLeg', etc.) MUST match the names in your avatar.glb file.
      // You may need to inspect your model in a tool like Blender to get the exact names.
      updateBoneRotation(11, 13, 15, 'LeftArm');
      updateBoneRotation(12, 14, 16, 'RightArm');
      updateBoneRotation(13, 11, 23, 'LeftForeArm');
      updateBoneRotation(14, 12, 24, 'RightForeArm');
      updateBoneRotation(23, 25, 27, 'LeftUpLeg');
      updateBoneRotation(24, 26, 28, 'RightUpLeg');
      updateBoneRotation(25, 27, 31, 'LeftLeg');
      updateBoneRotation(26, 28, 32, 'RightLeg');

      // More complex rotations (like the spine or head) would require more advanced calculations,
      // but this provides a solid foundation for the limbs.
    }
  });

  return <primitive object={scene} {...props} />;
}

// The main Avatar component that sets up the 3D canvas
export default function Avatar({ landmarks }) {
  return (
    <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px' }}>
      <Canvas camera={{ position: [0, 1.5, 3], fov: 50 }}>
        <ambientLight intensity={1.5} />
        <directionalLight position={[10, 10, 5]} intensity={2} />
        <Model landmarks={landmarks} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}
