import React from "react";

const PrimaryRoundedButton = ({ onClick, disabled, children, className }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`bg-blue-600 border-blue-600 border rounded-full inline-flex items-center justify-center py-3 px-7 text-base font-medium text-white 
                 hover:bg-blue-500 hover:border-blue-500 
                 disabled:bg-gray-300 disabled:border-gray-300 disabled:text-gray-500 
                 active:bg-blue-700 active:border-blue-700 transition-colors duration-200
                 ${className || ""}`}
    >
      {children}
    </button>
  );
};

export default PrimaryRoundedButton;
