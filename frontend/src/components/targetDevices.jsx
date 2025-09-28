import React, { useState } from "react";

const TargetDeviceSelector = ({ onSelect }) => {
  const devices = ["Android", "iOS", "Web", "Edge", "Server"];
  const [selected, setSelected] = useState("");

  const handleSelect = (device) => {
    setSelected(device);
    onSelect?.(device.toLowerCase());
  };

  return (
    <div className="mt-8 w-full max-w-3xl">
      {/* Section Title */}
      <h2 className="mb-4 text-xl font-semibold text-gray-900 dark:text-blue-800">
        2. Select Target Device
      </h2>

      {/* Devices Container */}
      <div className="flex flex-wrap gap-4">
        {devices.map((device) => (
          <div
            key={device}
            onClick={() => handleSelect(device)}
            className={`cursor-pointer px-6 py-4 rounded-lg border text-center font-medium transition-all duration-200
              ${selected === device
                ? "bg-blue-600 text-white border-blue-600 shadow-lg"
                : "bg-gray-50 text-gray-800 border-gray-300 hover:bg-blue-100 dark:bg-gray-700 dark:text-white dark:border-gray-600 dark:hover:bg-gray-600"
              }`}
          >
            {device}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TargetDeviceSelector;
