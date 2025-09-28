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
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* File Upload Section */}
          <FileUploader
            title="Upload ML Model"
            description="Supported formats: .h5, .pb, .pt, .pth, .onnx, .pkl"
            onChange={file => {
              setUploadedFile(file);
              setError(null);
            }}
          />

          {/* Device Selection */}
          <TargetDeviceSelector 
            onSelect={device => {
              setSelectedDevice(device);
              setError(null);
            }}
          />

          {/* Action Button */}
          <div className="mt-8 flex justify-center">
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
              className="mt-6 bg-red-50 border-red-200"
              title="Error"
              description={error}
            />
          )}

          {/* Results Display */}
          {optimizationResults && !error && (
            <div className="mt-8">
              <Card
                title="Optimization Results"
                description={
                  <div className="space-y-2">
                    <p>Original Size: {optimizationResults.metrics.original_size_mb.toFixed(2)} MB</p>
                    <p>Optimized Size: {optimizationResults.metrics.optimized_size_mb.toFixed(2)} MB</p>
                    <p>Size Reduction: {optimizationResults.metrics.size_reduction_percent.toFixed(1)}%</p>
                    <p>Original Latency: {optimizationResults.metrics.original_latency_ms.toFixed(2)} ms</p>
                    <p>Optimized Latency: {optimizationResults.metrics.optimized_latency_ms.toFixed(2)} ms</p>
                  </div>
                }
              />
              
              <div className="mt-4 flex justify-center">
                <PrimaryRoundedButton
                  onClick={() => window.location.href = `/api/download/${optimizationResults.optimized_model}`}
                  disabled={isOptimizing}
                >
                  Download Optimized Model
                </PrimaryRoundedButton>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
