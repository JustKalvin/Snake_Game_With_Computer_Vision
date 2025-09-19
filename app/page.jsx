"use client";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import Link from "next/link";

const App = () => {
  return (
    <div className="relative min-h-screen bg-sky-400 flex flex-col items-center overflow-hidden">

      {/* ===== HEADER AWAN ===== */}
      <header className="sticky top-0 w-full z-20 flex justify-center gap-8 py-2 bg-sky-400">
        {/* Awan 1 */}
        <img
          src="/img/cloud.png"
          alt="Cloud 1"
          className="w-32 h-auto object-contain opacity-90 animate-float-slow-right"
        />
        {/* Awan 2 */}
        <img
          src="/img/cloud.png"
          alt="Cloud 2"
          className="w-32 h-auto object-contain opacity-90 animate-float-slow-left"
        />
      </header>

      {/* ===== KONTEN UTAMA ===== */}
      <main className="z-30 flex flex-col items-center gap-4 mt-12">
        <div className="w-full max-w-lg">
          <DotLottieReact
            src="https://lottie.host/436ba80b-c8ec-4c55-bd40-5401c3ad2543/furEUhSyOl.lottie"
            loop
            autoplay
          />
        </div>

        <p className="text-white text-2xl font-bold drop-shadow-md">
          Welcome To Snake AI Game
        </p>
        <Link href={"/pages/snake"}>
          <button className="px-6 py-2 bg-green-600 text-white font-bold rounded-xl shadow-lg hover:bg-green-700 transition transform hover:scale-105">
            PLAY
          </button>
        </Link>
      </main>

      {/* ===== FOOTER RUMPUT ===== */}
      <footer className="absolute bottom-0 left-0 w-full z-30">
        <img
          src="/img/grass.png"
          alt="Grass"
          className="w-full object-cover"
        />
      </footer>
    </div>
  );
};

export default App;
