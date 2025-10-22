import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import PrimaryRoundedButton from './ui/button';

const Navbar = ({ onLogout }) => {
    const { currentUser } = useAuth();
    
    return (
        <div className="navbar sticky top-0 text-white bg-gradient-to-r from-sky-400 via-cyan-300 to-sky-600 shadow-md backdrop-blur-sm z-50 border-b border-black-700 w-full p-4 flex justify-between items-center">
            <h1 className='w-full font-bold text-xl backdrop-blur-2xl text-shadow-blue-600 text-white size-18/2 '>ModelFlex</h1>
            <nav className="flex items-center gap-4">
                {currentUser ? (
                    <>
                        <span className="text-sm">Welcome, {currentUser.email}</span>
                        <PrimaryRoundedButton
                            onClick={onLogout}
                            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 text-sm"
                        >
                            Logout
                        </PrimaryRoundedButton>
                    </>
                ) : (
                    <span className="text-sm">Please log in to continue</span>
                )}
            </nav>
        </div>
    );
};


export default Navbar;