import React, { useState } from "react";
import Navbar from "./components/navbar";
import FileUploader from "./components/fileuploader";
import PrimaryRoundedButton from "./components/ui/button";
import TargetDeviceSelector from "./components/targetDevices";
import Card from "./components/ui/card";

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    if (!uploadedFile || !selectedDevice) {
      setError("Please select both a file and a target device");
      return;
    }

    setError(null);
    setIsOptimizing(true);
    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("target_device", selectedDevice); // <-- pass in formData

    try {
      const response = await fetch("/api/optimize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Optimization failed: " + (await response.text()));
      }

      const result = await response.json();
      setOptimizationResults(result);
      setError(null);
    } catch (error) {
      console.error("Error during optimization:", error);
      setError(error.message);
    } finally {
      setIsOptimizing(false);
    }
  };



  return (
    <div className="App">
      <Navbar />
      <main
        className="flex flex-col items-center justify-center px-4 py-8 min-h-screen bg-gradient-to-br from-sky-300 via-cyan-50 to-sky-200"
        style={{ marginTop: "0px" }}
      >
        <div className="bg-gradient-to-br from-slate-100 via-slate-50 to-gray-100 rounded-2xl shadow-xl p-10 w-full max-w-4xl flex flex-col items-center">
          {/* File Upload Section */}
          <h2 className="text-2xl font-semibold mb-4 text-blue-900">Upload and Optimize Your Machine Learning Models</h2>
          <FileUploader
            title="Upload ML Model"
            description="Supported formats: .h5, .pb, .pt, .pth, .onnx, .pkl"
            onChange={(file) => {
              setUploadedFile(file);
              setError(null);
            }}
          />

          {/* Device Selection */}
          <TargetDeviceSelector
            onSelect={(device) => {
              setSelectedDevice(device);
              setError(null);
            }}
          />

          {/* Action Button */}
          <div className="mt-8">
            <PrimaryRoundedButton
              onClick={handleOptimize}
              disabled={!uploadedFile || !selectedDevice || isOptimizing}
            >
              {isOptimizing ? "Optimizing..." : "Optimize Model"}
            </PrimaryRoundedButton>
          </div>

          {/* Error Display */}
          {error && (
            <Card
              className="mt-6 bg-red-50 border border-red-200 text-center"
              title="Error"
              description={error}
            />
          )}

          {/* Results Display */}
          {optimizationResults && !error && (
            <div className="mt-8 w-full flex flex-col items-center">
              <Card
                title="Optimization Results"
                description={
                  <div className="space-y-2 text-center">
                    <p>Original Size: {optimizationResults.metrics.original_size_mb.toFixed(2)} MB</p>
                    <p>Optimized Size: {optimizationResults.metrics.optimized_size_mb.toFixed(2)} MB</p>
                    <p>Size Reduction: {optimizationResults.metrics.size_reduction_percent.toFixed(1)}%</p>
                    <p>Original Latency: {optimizationResults.metrics.original_latency_ms.toFixed(2)} ms</p>
                    <p>Optimized Latency: {optimizationResults.metrics.optimized_latency_ms.toFixed(2)} ms</p>
                  </div>
                }
              />

              <div className="mt-4">
                <PrimaryRoundedButton
                  onClick={() =>
                    (window.location.href = `/api/download/${optimizationResults.optimized_model}`)
                  }
                  disabled={isOptimizing}
                >
                  Download Optimized Model
                </PrimaryRoundedButton>
              </div>
            </div>
          )}
        </div>
      </main>

      <div className="fixed bottom-4 w-full text-center text-sm text-gray-600">
        &copy; 2024 ModelFlex. All rights reserved.
      </div>
    </div>
  );
}

export default App;
