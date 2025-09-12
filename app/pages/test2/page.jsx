"use client";

import { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

// Helper function to draw bounding boxes and labels
function drawDetection(ctx, detection, videoWidth, videoHeight) {
  const { x, y, w, h, score, classId } = detection;
  const labels = ["Bawah", "Atas"];
  const label = labels[Math.round(classId)];

  // Scale bounding box to video size
  const scaledX = x * videoWidth;
  const scaledY = y * videoHeight;
  const scaledW = w * videoWidth;
  const scaledH = h * videoHeight;

  // Draw the bounding box
  ctx.strokeStyle = "#00FF00"; // Green color for the box
  ctx.lineWidth = 2;
  ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

  // Draw the label background
  ctx.fillStyle = "#00FF00";
  const text = `${label} - ${Math.round(score * 100)}%`;
  const textWidth = ctx.measureText(text).width;
  ctx.fillRect(scaledX, scaledY > 20 ? scaledY : scaledY + scaledH, textWidth + 4, -20);

  // Draw the label text
  ctx.fillStyle = "#000000";
  ctx.fillText(text, scaledX + 2, scaledY > 20 ? scaledY - 6 : scaledY + scaledH - 6);
}


export default function YOLOPage() {
  const videoRef = useRef(null);
  const gameCanvasRef = useRef(null);
  const videoCanvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading...");
  const [position, setPosition] = useState({ x: 250, y: 200, isJumping: false });

  // 1. Load Model + Setup Webcam
  useEffect(() => {
    const loadAndSetup = async () => {
      try {
        const modelUrl = "/best_web_model/model.json";
        const loadedModel = await tf.loadGraphModel(modelUrl);
        setModel(loadedModel);
        console.log("✅ Model loaded!");

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (error) {
        console.error("Failed to load model or setup webcam:", error);
        setPrediction("Error: Gagal memuat model atau webcam.");
      }
    };

    loadAndSetup();
  }, []);

  // 2. Detection Loop
  useEffect(() => {
    if (!model || !videoRef.current) return;

    const detectFrame = async () => {
      if (videoRef.current.readyState < 4) {
        requestAnimationFrame(detectFrame);
        return;
      }

      const inputSize = 640;
      const scoreThreshold = 0.5;

      const detections = tf.tidy(() => {
        const tfImg = tf.browser.fromPixels(videoRef.current)
          .resizeBilinear([inputSize, inputSize])
          .toFloat()
          .div(255.0)
          .expandDims(0);

        // --- FIX STARTS HERE ---
        const outputTensor = model.execute(tfImg);

        // The model returns a single tensor of shape [1, N, 6]
        // where N is the number of detections, and 6 is [x_center, y_center, w, h, score, classId]
        const data = outputTensor.arraySync()[0];
        const detectedObjects = [];

        for (const det of data) {
          const [x_center, y_center, w, h, score, classId] = det;

          if (score > scoreThreshold) {
            // Convert [x_center, y_center, w, h] to [x_min, y_min, w, h] for drawing
            detectedObjects.push({
              x: x_center - w / 2,
              y: y_center - h / 2,
              w: w,
              h: h,
              score: score,
              classId: classId
            });
          }
        }
        return detectedObjects;
        // --- FIX ENDS HERE ---
      });

      if (videoCanvasRef.current) {
        const ctx = videoCanvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, videoCanvasRef.current.width, videoCanvasRef.current.height);
        ctx.drawImage(videoRef.current, 0, 0, videoCanvasRef.current.width, videoCanvasRef.current.height);

        detections.forEach(det => drawDetection(ctx, det, videoCanvasRef.current.width, videoCanvasRef.current.height));
      }

      if (detections.length > 0) {
        const bestDetection = detections.reduce((prev, current) =>
          (prev.score > current.score) ? prev : current
        );
        const labels = ["Bawah", "Atas"];
        const action = labels[Math.round(bestDetection.classId)];

        setPrediction(action);
        handleAction(action);
      } else {
        setPrediction("Tidak ada objek terdeteksi");
      }

      requestAnimationFrame(detectFrame);
    };

    videoRef.current.onloadeddata = () => {
      if (videoCanvasRef.current) {
        videoCanvasRef.current.width = videoRef.current.videoWidth;
        videoCanvasRef.current.height = videoRef.current.videoHeight;
      }
      detectFrame();
    };
  }, [model]);

  // 3. Game Action Logic
  const handleAction = (action) => {
    setPosition((prev) => {
      if (action === "Atas" && !prev.isJumping) {
        return { ...prev, isJumping: true, y: prev.y - 100 };
      }
      return prev;
    });

    if (action === "Atas") {
      setTimeout(() => {
        setPosition((p) => ({ ...p, y: 200, isJumping: false }));
      }, 500);
    }
  };

  // 4. Game Canvas Rendering
  useEffect(() => {
    if (!gameCanvasRef.current) return;
    const ctx = gameCanvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, gameCanvasRef.current.width, gameCanvasRef.current.height);
    ctx.fillStyle = "#4caf50";
    ctx.fillRect(0, 250, gameCanvasRef.current.width, 50);
    ctx.fillStyle = "#2196f3";
    ctx.fillRect(position.x, position.y, 50, 50);
  }, [position]);

  return (
    <div className="p-4 flex flex-col items-center gap-4">
      <h1 className="text-2xl font-bold">🎮 Game Gerakan</h1>
      <canvas
        ref={gameCanvasRef}
        width={600}
        height={300}
        className="border-2 border-gray-600 rounded-md bg-sky-200"
      />
      <div className="flex items-center gap-4">
        <video
          ref={videoRef}
          className="hidden"
          autoPlay
          muted
          playsInline
        />
        <canvas
          ref={videoCanvasRef}
          width="200"
          height="150"
          className="rounded-md border"
          style={{ transform: "scaleX(-1)" }}
        />
        <h2 className="text-lg font-semibold bg-gray-100 p-4 rounded-md">
          Prediksi: <span className="font-mono text-blue-600">{prediction}</span>
        </h2>
      </div>
    </div>
  );
}