import React from "react";

const Card = ({ title, description, onClick, className }) => {
  return (
    <div
      onClick={onClick}
      className={`cursor-pointer max-w-sm p-6 bg-white border border-gray-200 rounded-lg shadow-sm hover:bg-gray-100 transition-colors duration-300 dark:bg-grey-100 dark:border-gray-700-dotted dark:hover:bg-gray-200 ${className}`}
    >
      <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-400">
        {title}
      </h5>
      <p className="font-normal text-gray-700 dark:text-gray-400">
        {description}
      </p>
    </div>
  );
};

export default Card;
