"use client";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading...");
  const [position, setPosition] = useState({ x: 250, y: 200, isJumping: false });

  // Load model + setup webcam
  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel("/model/model.json");
      setModel(loadedModel);
      console.log("✅ Model loaded!");
    };
    loadModel();

    const setupCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    };
    setupCamera();
  }, []);

  // Prediksi
  const runPrediction = async () => {
    if (!model || !videoRef.current) return;

    const img = tf.browser.fromPixels(videoRef.current)
      .resizeNearestNeighbor([224, 224]) // sesuaikan dengan input model
      .toFloat()
      .expandDims(0);

    const output = model.predict(img);
    const predictions = await output.data();

    // 🔹 Label sesuai model kamu
    const labels = ["Kiri", "Kanan", "Atas", "Bawah", "Stop"];
    const maxIndex = predictions.indexOf(Math.max(...predictions));

    setPrediction(labels[maxIndex]);
    handleAction(labels[maxIndex]);

    tf.dispose(img);
  };

  // Aksi game sesuai prediksi
  const handleAction = (action) => {
    setPosition((prev) => {
      let newPos = { ...prev };
      if (action === "Kiri") newPos.x -= 10;
      if (action === "Kanan") newPos.x += 10;
      if (action === "Atas" && !prev.isJumping) {
        newPos.isJumping = true;
        newPos.y -= 50; // lompat ke atas
        setTimeout(() => {
          setPosition((p) => ({ ...p, y: 200, isJumping: false })); // kembali ke tanah
        }, 500);
      }
      if (action === "Bawah") {
        newPos.y += 10; // turun
        if (newPos.y > 200) newPos.y = 200; // jangan lebih rendah dari tanah
      }
      if (action === "Stop") {
        // diam di tempat, tidak ada perubahan posisi
      }
      return newPos;
    });
  };

  // Loop prediksi tiap 500ms
  useEffect(() => {
    const interval = setInterval(runPrediction, 500);
    return () => clearInterval(interval);
  }, [model]);

  // Render game (gambar karakter di canvas)
  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // gambar "tanah"
    ctx.fillStyle = "#4caf50";
    ctx.fillRect(0, 250, 600, 50);

    // gambar karakter (kotak biru)
    ctx.fillStyle = "#2196f3";
    ctx.fillRect(position.x, position.y, 50, 50);
  }, [position]);

  return (
    <div className="p-4 flex flex-col items-center">
      <h1 className="text-xl font-bold mb-2">🎮 Game Gerakan</h1>

      {/* Webcam */}
      <video
        ref={videoRef}
        width="200"
        height="150"
        className="rounded-md mb-4"
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Game Canvas */}
      <canvas ref={canvasRef} width={600} height={300} className="border border-gray-400 rounded-md" />

      {/* Prediction */}
      <h2 className="mt-4 text-lg">Prediksi: {prediction}</h2>
    </div>
  );
}
