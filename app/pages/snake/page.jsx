"use client";

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import React, { useState, useEffect, useRef } from "react";

// =======================================================================
// 🔹 KOMPONEN 1: DETEKTOR TANGAN (Dengan sedikit modifikasi)
// =======================================================================

const HAND_DETECTOR_CONFIG = {
  MODEL_PATH: "/best_web_model3/model.json",
  CLASS_NAMES: ["atas", "bawah", "kanan", "kiri"],
  CONFIDENCE_THRESHOLD: 0.5, // Naikkan sedikit threshold agar lebih stabil
};

function HandDetector({ onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  // 1. Muat model
  useEffect(() => {
    tf.setBackend("webgl");
    const loadModel = async () => {
      console.log("Memuat model deteksi...");
      try {
        const m = await tf.loadGraphModel(HAND_DETECTOR_CONFIG.MODEL_PATH);
        setModel(m);
        setLoading(false);
        console.log("✅ Model deteksi berhasil dimuat!");
      } catch (error) {
        console.error("Gagal memuat model deteksi:", error);
      }
    };
    loadModel();
  }, []);

  // 2. Mulai kamera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Gagal mengakses kamera:", err);
      }
    };
    startCamera();
  }, []);

  // 3. Loop Deteksi
  useEffect(() => {
    if (!model || !videoRef.current) return;
    let animationId;
    const detectFrame = async () => {
      if (videoRef.current && videoRef.current.readyState === 4) {
        const video = videoRef.current;
        tf.tidy(() => {
          const inputTensor = tf.browser.fromPixels(video).resizeBilinear([640, 640]).div(255.0).expandDims(0);
          const outputs = model.execute(inputTensor);
          const transposed = outputs.transpose([0, 2, 1]);
          const boxes = transposed.squeeze();
          const detections = boxes.arraySync();
          drawBoxes(detections, video, canvasRef.current);
        });
      }
      animationId = requestAnimationFrame(detectFrame);
    };
    detectFrame();
    return () => cancelAnimationFrame(animationId);
  }, [model]);

  // 4. Gambar Bounding Box
  const drawBoxes = (detections, video, canvas) => {
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let bestDetection = null;
    let maxOverallScore = 0;

    detections.forEach((det) => {
      const [...classScores] = det.slice(4);
      const maxScore = Math.max(...classScores);
      if (maxScore > HAND_DETECTOR_CONFIG.CONFIDENCE_THRESHOLD && maxScore > maxOverallScore) {
        maxOverallScore = maxScore;
        bestDetection = det;
      }
    });

    if (bestDetection) {
      const [x_center, y_center, w, h, ...classScores] = bestDetection;
      const classIndex = classScores.indexOf(Math.max(...classScores));
      const label = HAND_DETECTOR_CONFIG.CLASS_NAMES[classIndex];

      // 🚀 KIRIM PREDIKSI KE KOMPONEN INDUK
      onPrediction(label);

      const scaleX = video.videoWidth / 640;
      const scaleY = video.videoHeight / 640;
      const x1 = (x_center - w / 2) * scaleX;
      const y1 = (y_center - h / 2) * scaleY;
      const boxW = w * scaleX;
      const boxH = h * scaleY;

      ctx.strokeStyle = "lime";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxW, boxH);
      const text = `${label} (${(maxOverallScore * 100).toFixed(1)}%)`;
      ctx.fillStyle = "lime";
      ctx.font = "18px Arial";
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x1, y1 > 20 ? y1 - 22 : y1, textWidth + 10, 22);
      ctx.fillStyle = "black";
      ctx.fillText(text, x1 + 5, y1 > 20 ? y1 - 5 : y1 + 15);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold mb-2">
        {loading ? "⏳ Memuat Kamera..." : "✅ Kamera Siap"}
      </h2>
      <div className="relative border-4 border-gray-300 rounded-lg overflow-hidden">
        <video ref={videoRef} autoPlay playsInline muted width="480" height="360" style={{ transform: "scaleX(-1)" }} />
        <canvas ref={canvasRef} width="480" height="360" className="absolute top-0 left-0" style={{ transform: "scaleX(-1)" }} />
      </div>
    </div>
  );
}

// = a new helper function outside the component
// to draw rectangles with rounded corners.
function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.arcTo(x + width, y, x + width, y + radius, radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
  ctx.lineTo(x + radius, y + height);
  ctx.arcTo(x, y + height, x, y + height - radius, radius);
  ctx.lineTo(x, y + radius);
  ctx.arcTo(x, y, x + radius, y, radius);
  ctx.closePath();
  ctx.fill();
}


// =======================================================================
// 🔹 KOMPONEN 2: GAME SNAKE (DENGAN TAMPILAN BARU)
// =======================================================================

const GAME_CONFIG = {
  BOARD_SIZE_X: 17,
  BOARD_SIZE_Y: 15,
  GRID_SIZE: 20,
  INITIAL_SPEED: 200,
  SNAKE_BORDER_RADIUS: 6, // Atur radius di sini (misal: 10 untuk lingkaran penuh)
};

function SnakeGame({ direction }) {
  const canvasRef = useRef(null);

  // ... (semua logika state dan fungsi lain tetap sama)
  const generateFoodPosition = (snakeBody) => {
    let newFoodPosition;
    do {
      newFoodPosition = {
        x: Math.floor(Math.random() * GAME_CONFIG.BOARD_SIZE_X),
        y: Math.floor(Math.random() * GAME_CONFIG.BOARD_SIZE_Y),
      };
    } while (
      snakeBody.some(
        (segment) => segment.x === newFoodPosition.x && segment.y === newFoodPosition.y
      )
    );
    return newFoodPosition;
  };

  const getInitialSnake = () => [{ x: 8, y: 7 }];

  const [snake, setSnake] = useState(getInitialSnake);
  const [food, setFood] = useState(() => generateFoodPosition(getInitialSnake()));
  const [currentDirection, setCurrentDirection] = useState({ x: 0, y: -1 });
  const [gameOver, setGameOver] = useState(false);
  const [score, setScore] = useState(0);

  const resetGame = () => {
    const initialSnake = getInitialSnake();
    setSnake(initialSnake);
    setFood(generateFoodPosition(initialSnake));
    setCurrentDirection({ x: 0, y: -1 });
    setGameOver(false);
    setScore(0);
  };

  useEffect(() => {
    if (gameOver) return;
    const move = direction.toLowerCase();
    if (move === "atas" && currentDirection.y === 1) return;
    if (move === "bawah" && currentDirection.y === -1) return;
    if (move === "kiri" && currentDirection.x === 1) return;
    if (move === "kanan" && currentDirection.x === -1) return;
    switch (move) {
      case "atas": setCurrentDirection({ x: 0, y: -1 }); break;
      case "bawah": setCurrentDirection({ x: 0, y: 1 }); break;
      case "kiri": setCurrentDirection({ x: -1, y: 0 }); break;
      case "kanan": setCurrentDirection({ x: 1, y: 0 }); break;
      default: break;
    }
  }, [direction, gameOver]);

  useEffect(() => {
    if (gameOver) return;
    const gameInterval = setInterval(() => {
      setSnake((prevSnake) => {
        const newSnake = [...prevSnake];
        const head = { ...newSnake[0] };
        head.x += currentDirection.x;
        head.y += currentDirection.y;
        if (head.x < 0 || head.x >= GAME_CONFIG.BOARD_SIZE_X || head.y < 0 || head.y >= GAME_CONFIG.BOARD_SIZE_Y) {
          setGameOver(true);
          return prevSnake;
        }
        for (let i = 1; i < newSnake.length; i++) {
          if (head.x === newSnake[i].x && head.y === newSnake[i].y) {
            setGameOver(true);
            return prevSnake;
          }
        }
        newSnake.unshift(head);
        if (head.x === food.x && head.y === food.y) {
          setScore(s => s + 10);
          setFood(generateFoodPosition(newSnake));
        } else {
          newSnake.pop();
        }
        return newSnake;
      });
    }, GAME_CONFIG.INITIAL_SPEED);
    return () => clearInterval(gameInterval);
  }, [snake, currentDirection, food, gameOver]);

  // ✅ PERUBAHAN UTAMA ADA DI SINI (LOGIKA RENDER)
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const gridSize = GAME_CONFIG.GRID_SIZE;
    const radius = GAME_CONFIG.SNAKE_BORDER_RADIUS;

    // 1. Gambar latar belakang (pola grid) - tetap sama
    for (let row = 0; row < GAME_CONFIG.BOARD_SIZE_Y; row++) {
      for (let col = 0; col < GAME_CONFIG.BOARD_SIZE_X; col++) {
        ctx.fillStyle = (row + col) % 2 === 0 ? "#A3D866" : "#83C94D";
        ctx.fillRect(col * gridSize, row * gridSize, gridSize, gridSize);
      }
    }

    // 2. Gambar ular dengan tampilan baru
    snake.forEach((segment, index) => {
      const x = segment.x * gridSize;
      const y = segment.y * gridSize;

      if (index === 0) {
        // KEPALA ULAR: Buat gradient berdasarkan arah gerak
        let gradient;
        if (currentDirection.x === 1) { // Kanan
          gradient = ctx.createLinearGradient(x, y, x + gridSize, y);
        } else if (currentDirection.x === -1) { // Kiri
          gradient = ctx.createLinearGradient(x + gridSize, y, x, y);
        } else if (currentDirection.y === 1) { // Bawah
          gradient = ctx.createLinearGradient(x, y, x, y + gridSize);
        } else { // Atas
          gradient = ctx.createLinearGradient(x, y + gridSize, x, y);
        }

        gradient.addColorStop(0, "#5E9EFF"); // Warna gradient awal
        gradient.addColorStop(1, "#2F65C0"); // Warna gradient akhir
        ctx.fillStyle = gradient;

      } else {
        // BADAN ULAR: Warna solid
        ctx.fillStyle = "#3668A6";
      }

      // Gambar segmen menggunakan fungsi baru
      drawRoundedRect(ctx, x + 1, y + 1, gridSize - 2, gridSize - 2, radius);
    });

    // 3. Gambar makanan dengan tampilan baru (opsional)
    ctx.fillStyle = "#E86F6F";
    drawRoundedRect(ctx, food.x * gridSize + 1, food.y * gridSize + 1, gridSize - 2, gridSize - 2, radius);

    // TAMBAHKAN `currentDirection` karena gradient bergantung padanya
  }, [snake, food, currentDirection]);

  // Return JSX (tetap sama)
  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold mb-2">Skor: {score}</h2>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={GAME_CONFIG.BOARD_SIZE_X * GAME_CONFIG.GRID_SIZE}
          height={GAME_CONFIG.BOARD_SIZE_Y * GAME_CONFIG.GRID_SIZE}
          className="bg-gray-800 border-4 border-gray-300 rounded-lg"
        />
        {gameOver && (
          <div className="absolute inset-0 flex flex-col justify-center items-center bg-black bg-opacity-70">
            <h3 className="text-4xl text-white font-bold">GAME OVER</h3>
            <button
              onClick={resetGame}
              className="mt-4 px-4 py-2 bg-green-500 text-white font-bold rounded hover:bg-green-600"
            >
              Mulai Lagi
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// =======================================================================
// 🔹 KOMPONEN 3: HALAMAN UTAMA (Induk)
// =======================================================================

export default function GamePage() {
  const [lastPrediction, setLastPrediction] = useState("ATAS");

  // Fungsi callback yang akan menerima prediksi dari HandDetector
  const handlePrediction = (prediction) => {
    console.log("Gerakan terdeteksi:", prediction);
    setLastPrediction(prediction);
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-4xl font-extrabold mb-6 text-gray-800">Snake AI 🐍</h1>
      <p className="mb-8 text-gray-600">Gunakan gerakan tangan untuk mengontrol ular!</p>

      <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl justify-center items-start">
        {/* Sisi Kiri: Game Snake */}
        <div className="w-full md:w-auto">
          <SnakeGame direction={lastPrediction} />
        </div>

        {/* Sisi Kanan: Tampilan Kamera */}
        <div className="w-full md:w-auto">
          <HandDetector onPrediction={handlePrediction} />
          <div className="mt-4 p-4 bg-white rounded-lg shadow-md w-[480px] text-center">
            <h3 className="font-semibold text-lg">Perintah Terakhir:</h3>
            <p className="text-2xl font-bold text-red-600">{lastPrediction}</p>
          </div>
        </div>
      </div>
    </main>
  );
}