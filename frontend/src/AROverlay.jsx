import React, { useRef, useEffect } from 'react';

/**
 * AROverlay Component - Dynamic AR Visualization
 * 
 * Provides real-time visual feedback through:
 * - Skeleton rendering with colored joints
 * - Target "shadow" poses for alignment
 * - Directional arrows pointing to ideal positions
 * - Colored angle sectors (green = correct, red = error)
 * 
 * Based on FitCoachAR proposal Section 4.4
 */
export default function AROverlay({ 
  landmarks, 
  feedbackLandmarks = [],
  selectedExercise,
  targetAngles = {},
  currentAngles = {}
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !landmarks || landmarks.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // MediaPipe Pose connections for skeleton rendering
    const POSE_CONNECTIONS = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms
      [11, 23], [12, 24], [23, 24], // Torso
      [23, 25], [25, 27], [27, 29], [27, 31], // Left leg
      [24, 26], [26, 28], [28, 30], [28, 32], // Right leg
    ];

    // Draw skeleton connections
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 3;
    POSE_CONNECTIONS.forEach(([start, end]) => {
      const startLm = landmarks[start];
      const endLm = landmarks[end];
      if (startLm && endLm && startLm.visibility > 0.5 && endLm.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(startLm.x * canvas.width, startLm.y * canvas.height);
        ctx.lineTo(endLm.x * canvas.width, endLm.y * canvas.height);
        ctx.stroke();
      }
    });

    // Draw landmarks with color-coded feedback
    landmarks.forEach((lm, index) => {
      if (lm.visibility < 0.5) return;

      const x = lm.x * canvas.width;
      const y = lm.y * canvas.height;

      // Check if this landmark has feedback
      const hasError = feedbackLandmarks.includes(index);

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = hasError ? '#ef4444' : '#10b981'; // Red for error, green for good
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw angle indicators for relevant joints
    const drawAngleIndicator = (centerIdx, angle, targetAngle, label) => {
      const lm = landmarks[centerIdx];
      if (!lm || lm.visibility < 0.5) return;

      const x = lm.x * canvas.width;
      const y = lm.y * canvas.height;
      const radius = 40;

      // Determine if angle is within acceptable range (±15 degrees)
      const isCorrect = Math.abs(angle - targetAngle) < 15;
      const color = isCorrect ? '#10b981' : '#ef4444';

      // Draw angle arc
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, (angle / 180) * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw label
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(`${Math.round(angle)}°`, x + radius + 5, y);
    };

    // Exercise-specific angle visualization
    if (selectedExercise === 'bicep_curls') {
      if (currentAngles.rightElbow !== undefined && targetAngles.rightElbow !== undefined) {
        drawAngleIndicator(14, currentAngles.rightElbow, targetAngles.rightElbow, 'R Elbow');
      }
    } else if (selectedExercise === 'squats') {
      if (currentAngles.rightKnee !== undefined && targetAngles.rightKnee !== undefined) {
        drawAngleIndicator(26, currentAngles.rightKnee, targetAngles.rightKnee, 'R Knee');
      }
    }

    // Draw directional arrows for correction guidance
    feedbackLandmarks.forEach(idx => {
      const lm = landmarks[idx];
      if (!lm || lm.visibility < 0.5) return;

      const x = lm.x * canvas.width;
      const y = lm.y * canvas.height;

      // Draw arrow based on exercise and joint
      let arrowDirection = { dx: 0, dy: 0 };
      
      if (selectedExercise === 'bicep_curls' && idx === 14) { // Right elbow
        arrowDirection = { dx: 0, dy: -30 }; // Point upward for curl
      } else if (selectedExercise === 'squats' && idx === 26) { // Right knee
        arrowDirection = { dx: 0, dy: 30 }; // Point downward for depth
      }

      if (arrowDirection.dx !== 0 || arrowDirection.dy !== 0) {
        ctx.strokeStyle = '#facc15'; // Yellow arrow
        ctx.fillStyle = '#facc15';
        ctx.lineWidth = 3;

        // Arrow shaft
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + arrowDirection.dx, y + arrowDirection.dy);
        ctx.stroke();

        // Arrow head
        const headLength = 10;
        const angle = Math.atan2(arrowDirection.dy, arrowDirection.dx);
        ctx.beginPath();
        ctx.moveTo(x + arrowDirection.dx, y + arrowDirection.dy);
        ctx.lineTo(
          x + arrowDirection.dx - headLength * Math.cos(angle - Math.PI / 6),
          y + arrowDirection.dy - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          x + arrowDirection.dx - headLength * Math.cos(angle + Math.PI / 6),
          y + arrowDirection.dy - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      }
    });

  }, [landmarks, feedbackLandmarks, selectedExercise, targetAngles, currentAngles]);

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={480}
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        zIndex: 2,
        pointerEvents: 'none'
      }}
    />
  );
}
