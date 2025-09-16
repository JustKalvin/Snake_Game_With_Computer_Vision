"use client";

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import React, { useState, useEffect, useRef } from "react";

// =======================================================================
// 🔹 KONFIGURASI & FUNGSI BANTUAN
// =======================================================================

const HAND_DETECTOR_CONFIG = {
  MODEL_PATH: "/best_web_model8/model.json",
  CLASS_NAMES: [
    "Help Me",
    "kiri",
    "kanan",
    "Level 3",
    "Level 4",
    "Level 5",
    "atas",
    "bawah"
  ],
  CONFIDENCE_THRESHOLD: 0.25,
};

// UKURAN BARU: Grid size diperbesar dari 20 menjadi 28
const GAME_CONFIG = {
  BOARD_SIZE_X: 17,
  BOARD_SIZE_Y: 15,
  GRID_SIZE: 28, // <-- DIPERBESAR
  INITIAL_SPEED: 500,
  SNAKE_BORDER_RADIUS: 8, // <-- Disesuaikan
};

// Fungsi bantuan untuk menggambar sudut tumpul
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
// 🔹 KOMPONEN 1: DETEKTOR TANGAN (DENGAN PERBAIKAN)
// =======================================================================

function HandDetector({ onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    tf.setBackend("webgl");
    const loadModel = async () => {
      try {
        const m = await tf.loadGraphModel(HAND_DETECTOR_CONFIG.MODEL_PATH);
        setModel(m);
        setLoading(false);
      } catch (error) {
        console.error("Gagal memuat model deteksi:", error);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
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

  // ✅ KEMBALIKAN KE LOGIKA YANG BENAR
  useEffect(() => {
    if (!model || !videoRef.current) return;
    let animationId;
    const detectFrame = async () => {
      if (videoRef.current && videoRef.current.readyState === 4) {
        const video = videoRef.current;
        tf.tidy(() => {
          const inputTensor = tf.browser.fromPixels(video).resizeBilinear([640, 640]).div(255.0).expandDims(0);

          // Logika pemrosesan output model yang benar untuk model ini
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

  // ✅ KEMBALIKAN FUNGSI drawBoxes KE VERSI YANG BENAR
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

      onPrediction(label);

      const scaleX = video.videoWidth / 640;
      const scaleY = video.videoHeight / 640;
      const x1 = (x_center - w / 2) * scaleX;
      const y1 = (y_center - h / 2) * scaleY;
      const boxW = w * scaleX;
      const boxH = h * scaleY;

      ctx.strokeStyle = "#10B981"; // Emerald-500
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, boxW, boxH);

      const text = `${label} (${(maxOverallScore * 100).toFixed(1)}%)`;
      ctx.fillStyle = "#10B981";
      ctx.font = "18px sans-serif";
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x1 - 2, y1 > 20 ? y1 - 22 : y1, textWidth + 12, 22);
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(text, x1 + 5, y1 > 20 ? y1 - 5 : y1 + 15);
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto">
      <h2 className="text-2xl font-bold text-white mb-4 text-center">
        {loading ? "⏳ Memuat Kamera..." : "✅ Kamera Siap"}
      </h2>
      <div className="relative aspect-video rounded-2xl overflow-hidden shadow-2xl border-4 border-gray-700 bg-gray-900">
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ transform: "scaleX(-1)" }} />
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" style={{ transform: "scaleX(-1)" }} />
      </div>
    </div>
  );
}


// =======================================================================
// 🔹 KOMPONEN 2: GAME SNAKE (DENGAN WARNA BARU)
// =======================================================================

function SnakeGame({ direction }) {
  const canvasRef = useRef(null);
  const [snake, setSnake] = useState(() => [{ x: 8, y: 7 }]);
  const [food, setFood] = useState(() => generateFoodPosition([{ x: 8, y: 7 }]));
  const [currentDirection, setCurrentDirection] = useState({ x: 0, y: -1 });
  const [gameOver, setGameOver] = useState(false);
  const [score, setScore] = useState(0);
  const [isScoreAnimating, setIsScoreAnimating] = useState(false);

  useEffect(() => {
    if (score > 0) {
      setIsScoreAnimating(true);
      const timer = setTimeout(() => setIsScoreAnimating(false), 200);
      return () => clearTimeout(timer);
    }
  }, [score]);

  const resetGame = () => {
    const initialSnake = [{ x: 8, y: 7 }];
    setSnake(initialSnake);
    setFood(generateFoodPosition(initialSnake));
    setCurrentDirection({ x: 0, y: -1 });
    setGameOver(false);
    setScore(0);
  };

  function generateFoodPosition(snakeBody) {
    let newFoodPosition;
    do {
      newFoodPosition = {
        x: Math.floor(Math.random() * GAME_CONFIG.BOARD_SIZE_X),
        y: Math.floor(Math.random() * GAME_CONFIG.BOARD_SIZE_Y),
      };
    } while (snakeBody.some(segment => segment.x === newFoodPosition.x && segment.y === newFoodPosition.y));
    return newFoodPosition;
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
      setSnake(prevSnake => {
        const newSnake = [...prevSnake];
        const head = { ...newSnake[0] };
        head.x += currentDirection.x;
        head.y += currentDirection.y;

        if (
          head.x < 0 || head.x >= GAME_CONFIG.BOARD_SIZE_X ||
          head.y < 0 || head.y >= GAME_CONFIG.BOARD_SIZE_Y ||
          newSnake.some(segment => segment.x === head.x && segment.y === head.y)
        ) {
          setGameOver(true);
          return prevSnake;
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
  }, [currentDirection, food, gameOver]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const { GRID_SIZE, SNAKE_BORDER_RADIUS } = GAME_CONFIG;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < GAME_CONFIG.BOARD_SIZE_Y; row++) {
      for (let col = 0; col < GAME_CONFIG.BOARD_SIZE_X; col++) {
        ctx.fillStyle = (row + col) % 2 === 0 ? "#2D3748" : "#1A202C";
        ctx.fillRect(col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE);
      }
    }

    snake.forEach((segment, index) => {
      const x = segment.x * GRID_SIZE, y = segment.y * GRID_SIZE;
      if (index === 0) {
        let gradient;
        if (currentDirection.x === 1) gradient = ctx.createLinearGradient(x, y, x + GRID_SIZE, y);
        else if (currentDirection.x === -1) gradient = ctx.createLinearGradient(x + GRID_SIZE, y, x, y);
        else if (currentDirection.y === 1) gradient = ctx.createLinearGradient(x, y, x, y + GRID_SIZE);
        else gradient = ctx.createLinearGradient(x, y + GRID_SIZE, x, y);
        gradient.addColorStop(0, "#38B2AC"); gradient.addColorStop(1, "#319795");
        ctx.fillStyle = gradient;
      } else {
        // ✅ WARNA BARU UNTUK BADAN ULAR
        ctx.fillStyle = "#2C7A7B"; // Warna Teal lebih gelap
      }
      drawRoundedRect(ctx, x + 2, y + 2, GRID_SIZE - 4, GRID_SIZE - 4, SNAKE_BORDER_RADIUS);
    });

    ctx.fillStyle = "#F56565";
    drawRoundedRect(ctx, food.x * GRID_SIZE + 2, food.y * GRID_SIZE + 2, GRID_SIZE - 4, GRID_SIZE - 4, SNAKE_BORDER_RADIUS);
  }, [snake, food, currentDirection]);

  // JSX return tidak berubah
  return (
    <div className="w-full max-w-lg mx-auto flex flex-col items-center">
      <div className={`mb-4 text-center bg-gray-900/50 backdrop-blur-sm p-4 rounded-xl shadow-lg transition-transform duration-200 ${isScoreAnimating ? 'scale-110' : 'scale-100'}`}>
        <span className="text-lg font-semibold text-gray-300">SKOR</span>
        <p className="text-5xl font-bold text-white tracking-wider">{score}</p>
      </div>
      <div className="relative rounded-2xl overflow-hidden shadow-2xl border-4 border-gray-700">
        <canvas
          ref={canvasRef}
          width={GAME_CONFIG.BOARD_SIZE_X * GAME_CONFIG.GRID_SIZE}
          height={GAME_CONFIG.BOARD_SIZE_Y * GAME_CONFIG.GRID_SIZE}
        />
        {gameOver && (
          <div className="absolute inset-0 flex flex-col justify-center items-center bg-black/70 backdrop-blur-sm animate-fade-in">
            <h3 className="text-6xl text-white font-extrabold tracking-tighter animate-pop-in" style={{ "--delay": "100ms" }}>GAME OVER</h3>
            <button
              onClick={resetGame}
              className="mt-6 px-6 py-3 bg-teal-500 text-white font-bold rounded-lg shadow-lg hover:bg-teal-400 focus:outline-none focus:ring-4 focus:ring-teal-300/50 transition-all duration-300 transform hover:scale-105 active:scale-95 animate-pop-in"
              style={{ "--delay": "250ms" }}
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
// 🔹 KOMPONEN 3: HALAMAN UTAMA (GamePage)
// =======================================================================

export default function GamePage() {
  const [lastPrediction, setLastPrediction] = useState("ATAS");
  const handlePrediction = (prediction) => {
    if (prediction !== lastPrediction) {
      setLastPrediction(prediction);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen w-full bg-gray-900 text-white p-4 lg:p-8 overflow-hidden">
      {/* Efek Latar Belakang */}
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-gray-900 via-gray-800 to-black z-0"></div>
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-teal-500/10 rounded-full filter blur-3xl animate-pulse-slow"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-red-500/10 rounded-full filter blur-3xl animate-pulse-slow animation-delay-4000"></div>

      <div className="relative z-10 w-full flex flex-col items-center">
        <header className="text-center mb-8 animate-fade-in">
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-emerald-500">
            Snake AI 🐍
          </h1>
          <p className="mt-2 text-lg text-gray-400">
            Kontrol ular dengan gerakan tangan Anda!
          </p>
        </header>

        {/* Kontainer Utama */}
        <div className="flex flex-col lg:flex-row gap-8 w-full max-w-7xl justify-center items-start">

          {/* Sisi Kiri: Game */}
          <section className="w-full lg:w-1/2 animate-slide-in-left">
            <SnakeGame direction={lastPrediction} />
          </section>

          {/* Sisi Kanan: Kamera & Info */}
          <section className="w-full lg:w-1/2 flex flex-col gap-8 animate-slide-in-right">
            <HandDetector onPrediction={handlePrediction} />
            <div className="w-full max-w-lg mx-auto bg-gray-900/50 backdrop-blur-sm p-6 rounded-2xl shadow-lg border border-gray-700 text-center">
              <h3 className="font-semibold text-lg text-gray-300">PERINTAH TERDETEKSI</h3>
              <p className="text-4xl font-bold text-teal-400 tracking-wider mt-2">{lastPrediction.toUpperCase()}</p>
            </div>
          </section>

        </div>
      </div>
    </main>
  );
}