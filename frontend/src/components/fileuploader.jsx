import React, { useState } from "react";
import Card from "./ui/card"; // make sure path is correct

const FileUploader = ({ title, description, onChange }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    onChange?.(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    onChange?.(droppedFile);
  };

  return (
    <div className="mt-8 w-full max-w-3xl bg-center flex-items-center flex-col">
      {/* Section Title */}
      <div className="mb-4 text-xl font-semibold text-gray-900 dark:text-blue-800">
        1. Upload Model
      </div>

      {/* File Upload Card */}
      <div
        className="align-top bg-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <label className="block cursor-pointer">
          <Card
            title={file ? file.name : title}
            description={file ? "File ready to upload" : description}
            className="hover:bg-gray-100 transition-colors duration-300"
          />
          <input type="file" className="hidden" onChange={handleFileChange} />
        </label>
      </div>
    </div>
  );
};

export default FileUploader;
