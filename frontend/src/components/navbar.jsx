import React from 'react';
const Navbar = () => {
    return (
        <div className="navbar sticky top-0 text-white bg-gradient-to-r from-sky-400 via-cyan-300 to-sky-600 shadow-md backdrop-blur-sm z-50 border-b border-black-700 w-full p-4 flex justify-between items-center">
            <h1 className='w-full font-bold text-xl backdrop-blur-2xl text-shadow-blue-600 text-white size-18/2 '>ModelFlex</h1>
            <nav>
                <ul className="flex gap-8">
                    <li className="relative group cursor-pointer inline-block">
                        <span>Upload</span>
                        <span className="absolute -bottom-1 left-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                        <span className="absolute -bottom-1 right-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                    </li>
                    <li className="relative group cursor-pointer inline-block">
                        <span>Optimize</span>
                        <span className="absolute -bottom-1 left-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                        <span className="absolute -bottom-1 right-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                    </li>
                    <li className="relative group cursor-pointer inline-block">
                        <span>Download</span>
                        <span className="absolute -bottom-1 left-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                        <span className="absolute -bottom-1 right-1/2 w-0 transition-all h-0.5 bg-white group-hover:w-3/6"></span>
                    </li>
                </ul>


            </nav>
        </div>

    );
};


export default Navbar;