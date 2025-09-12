"use client";



import * as tf from "@tensorflow/tfjs";

import "@tensorflow/tfjs-backend-webgl";

import { useEffect, useRef, useState } from "react";



// --- KONFIGURASI ---

const MODEL_PATH = "/best_web_model2/model.json";

const CLASS_NAMES = ["KANAN", "KIRI", "BAWAH", "ATAS"]; // Ganti dengan nama kelas Anda

const CONFIDENCE_THRESHOLD = 0.05; // Ambang batas kepercayaan



export default function HandDetector() {

  const videoRef = useRef(null);

  const canvasRef = useRef(null);

  const [model, setModel] = useState(null);

  const [loading, setLoading] = useState(true);

  const [predictions, setPredictions] = useState([]);



  // 1. Muat model

  useEffect(() => {

    tf.setBackend("webgl");

    const loadModel = async () => {

      console.log("Memuat model...");

      try {

        const m = await tf.loadGraphModel(MODEL_PATH);

        setModel(m);

        setLoading(false);

        console.log("✅ Model berhasil dimuat!");

      } catch (error) {

        console.error("Gagal memuat model:", error);

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

      if (videoRef.current.readyState === 4) {

        const video = videoRef.current;



        tf.tidy(() => {

          const inputTensor = tf.browser

            .fromPixels(video)

            .resizeBilinear([640, 640])

            .div(255.0)

            .expandDims(0);



          const outputs = model.execute(inputTensor); // Shape: [1, 8, 8400]



          const transposed = outputs.transpose([0, 2, 1]);

          const boxes = transposed.squeeze(); // Shape: [8400, 8]



          const detections = boxes.arraySync();



          // 🔹 Proses prediksi + gambar

          const predList = drawBoxes(detections, video, canvasRef.current);

          setPredictions(predList); // Simpan ke state agar bisa ditampilkan

        });

      }

      animationId = requestAnimationFrame(detectFrame);

    };



    detectFrame();



    return () => cancelAnimationFrame(animationId);

  }, [model]);



  // 4. Gambar Bounding Box + return hasil prediksi

  const drawBoxes = (detections, video, canvas) => {

    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);



    const scaleX = video.videoWidth / 640;

    const scaleY = video.videoHeight / 640;



    let bestDetection = null;

    let maxOverallScore = 0;



    // Cari deteksi terbaik berdasarkan skor tertinggi

    detections.forEach((det) => {

      const [x_center, y_center, w, h, ...classScores] = det;



      let maxScore = 0;

      classScores.forEach((score) => {

        if (score > maxScore) {

          maxScore = score;

        }

      });



      if (maxScore > CONFIDENCE_THRESHOLD && maxScore > maxOverallScore) {

        maxOverallScore = maxScore;

        bestDetection = det;

      }

    });



    let predList = [];



    if (bestDetection) {

      const [x_center, y_center, w, h, ...classScores] = bestDetection;



      let maxScore = 0;

      let classIndex = -1;

      classScores.forEach((score, i) => {

        if (score > maxScore) {

          maxScore = score;

          classIndex = i;

        }

      });



      const x1 = (x_center - w / 2) * scaleX;

      const y1 = (y_center - h / 2) * scaleY;

      const boxW = w * scaleX;

      const boxH = h * scaleY;



      // Warna bounding box

      ctx.strokeStyle = "red"; // Ganti dari 'lime' ke 'red'

      ctx.lineWidth = 2;

      ctx.strokeRect(x1, y1, boxW, boxH);



      const label = `${CLASS_NAMES[classIndex]} (${(maxScore * 100).toFixed(

        1

      )}%)`;



      // Warna latar belakang label (masih hijau agar kontras dengan teks merah)

      ctx.fillStyle = "lime";

      ctx.font = "16px Arial";

      const textWidth = ctx.measureText(label).width;

      ctx.fillRect(x1, y1 > 20 ? y1 - 20 : y1, textWidth + 8, 20);



      // Warna teks label (merah)

      ctx.fillStyle = "black"; // Ganti dari 'black' ke 'red'

      ctx.fillText(label, x1 + 4, y1 > 20 ? y1 - 5 : y1 + 15);



      predList.push(label);

    }



    return predList;

  };



  return (

    <div className="flex flex-col items-center p-4">

      <h1 className="text-2xl font-bold mb-4">

        {loading ? "⏳ Memuat Model..." : "✅ Model Siap"}

      </h1>

      <div className="relative">

        <video

          ref={videoRef}

          autoPlay

          playsInline

          muted

          width="640"

          height="480"

          className="rounded-lg shadow-lg"

          style={{ transform: "scaleX(-1)" }}

        />

        <canvas

          ref={canvasRef}

          width="640"

          height="480"

          className="absolute top-0 left-0"

          style={{ transform: "scaleX(-1)" }}

        />

      </div>



      {/* 🔹 Hasil Prediksi */}

      <div className="mt-4 w-full max-w-md bg-gray-100 p-3 rounded-lg shadow">

        <h2 className="text-lg text-red-600 font-semibold mb-2">Hasil Prediksi:</h2>

        {predictions.length === 0 ? (

          <p className="text-red-500">Tidak ada deteksi...</p>

        ) : (

          <ul className="list-disc pl-5">

            {predictions.map((p, idx) => (

              <li key={idx} className="text-red-700 font-bold">{p}</li>

            ))} {/* ✅ INI BAGIAN YANG DIPERBAIKI */}

          </ul>

        )}

      </div>

    </div>

  );

}