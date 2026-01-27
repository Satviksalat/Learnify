import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = ({ toggleSidebar }) => {
    return (
        <nav className="navbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <button className="menu-btn" onClick={toggleSidebar}>☰</button>
                <Link to="/" className="logo">Learnify</Link>
            </div>
            <div className="nav-links">
                <Link to="/">Home</Link>
                <Link to="/tutorial/unit1-intro">Tutorials</Link>
                <Link to="/examples">Examples</Link>
                <Link to="/exercises">Exercises</Link>
                <Link to="/editor">Compiler</Link>
                <Link to="/quizzes">Quizzes</Link>
                <Link to="/resources">Resources</Link>
            </div>
        </nav>
    );
};

export default Navbar;
