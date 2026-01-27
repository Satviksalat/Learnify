import React from 'react';
import { Link } from 'react-router-dom';
import { getCompletedTutorials } from '../utils/progress';

const HomePage = () => {
    return (
        <div className="home-page" style={{ padding: '40px', maxWidth: '900px', margin: '0 auto' }}>
            <div className="hero-section" style={{ textAlign: 'center', marginBottom: '60px' }}>
                <h1 style={{ fontSize: '3em', marginBottom: '20px', color: '#333' }}>Learn Python Programming</h1>
                <p style={{ fontSize: '1.2em', color: '#666', marginBottom: '30px' }}>
                    Master the world's most popular programming language with our comprehensive, step-by-step interactive course.
                </p>
                {/* Resume Learning Button */}
                {(() => {
                    const last = localStorage.getItem('lastTutorial');
                    if (last) {
                        try {
                            const { id, title } = JSON.parse(last);
                            return (
                                <div style={{ marginBottom: '20px' }}>
                                    <Link to={`/tutorial/${id}`} style={{
                                        backgroundColor: '#ffc107',
                                        color: '#333',
                                        padding: '15px 40px',
                                        borderRadius: '30px',
                                        textDecoration: 'none',
                                        fontSize: '1.3em',
                                        fontWeight: 'bold',
                                        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                                        display: 'inline-block'
                                    }}>
                                        ▶ Resume: {title}
                                    </Link>
                                </div>
                            );
                        } catch (e) { return null; }
                    }
                })()}

                <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
                    <Link to="/tutorial/unit1-intro" style={{
                        backgroundColor: '#007bff',
                        color: 'white',
                        padding: '15px 30px',
                        borderRadius: '5px',
                        textDecoration: 'none',
                        fontSize: '1.2em',
                        fontWeight: 'bold'
                    }}>
                        Start Python Course
                    </Link>
                    <Link to="/tutorial/ml-unit1-intro" style={{
                        backgroundColor: '#28a745',
                        color: 'white',
                        padding: '15px 30px',
                        borderRadius: '5px',
                        textDecoration: 'none',
                        fontSize: '1.2em',
                        fontWeight: 'bold'
                    }}>
                        Start ML Course
                    </Link>
                </div>

                {/* Progress Summary */}
                <div style={{ marginTop: '30px', padding: '15px', backgroundColor: '#e9ecef', borderRadius: '8px', display: 'inline-block' }}>
                    <span style={{ fontSize: '1.1em', color: '#555' }}>
                        Your Progress: <strong>{getCompletedTutorials().length}</strong> tutorials completed
                    </span>
                </div>
            </div>

            <div className="info-section">
                <h2>Why Learn Python?</h2>
                <ul style={{ lineHeight: '1.6', color: '#444' }}>
                    <li><strong>Easy to Read:</strong> Simple syntax that mimics natural language.</li>
                    <li><strong>Versatile:</strong> Used in Web Dev, Data Science, AI, Automation, and more.</li>
                    <li><strong>High Demand:</strong> One of the most sought-after skills in the job market.</li>
                </ul>
            </div>

            <div className="info-section" style={{ marginTop: '40px' }}>
                <h2>Course Objectives</h2>
                <p style={{ lineHeight: '1.6', color: '#444' }}>
                    By the end of this course, you will be able to:
                </p>
                <ul style={{ lineHeight: '1.6', color: '#444' }}>
                    <li>Write confident Python code using core concepts.</li>
                    <li>Understand Object-Oriented Programming (OOP) principles.</li>
                    <li>Create visualizations and solve algorithmic problems.</li>
                    <li>Build networked applications and GUIs.</li>
                    <li>Connect and interact with Database systems (MySQL).</li>
                </ul>
            </div>

            <div className="info-section" style={{ marginTop: '40px' }}>
                <h2>Syllabus Overview</h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
                    <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
                        <h3>Unit 1: Basics</h3>
                        <p>Syntax, Variables, Loops, Functions</p>
                    </div>
                    <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
                        <h3>Unit 2: OOP</h3>
                        <p>Classes, Objects, Inheritance</p>
                    </div>
                    <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
                        <h3>Unit 3: Algorithms</h3>
                        <p>Plotting, DP, Optimization</p>
                    </div>
                    <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
                        <h3>Unit 4: Network & GUI</h3>
                        <p>Sockets, Emails, Tkinter</p>
                    </div>
                    <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
                        <h3>Unit 5: Database</h3>
                        <p>MySQL, CRUD operations</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
