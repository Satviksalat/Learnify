import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const ExamplesPage = () => {
    const [programs, setPrograms] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('Python'); // 'Python' or 'Machine Learning'

    useEffect(() => {
        const apiUrl = import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com';
        fetch(`${apiUrl}/api/programs`)
            .then(res => res.json())
            .then(data => {
                setPrograms(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error fetching programs:", err);
                setLoading(false);
            });
    }, []);

    // Filter and Group Data
    const filteredPrograms = programs.filter(p => p.course === activeTab);

    // Group by Unit
    const groupedPrograms = filteredPrograms.reduce((acc, curr) => {
        if (!acc[curr.unit]) {
            acc[curr.unit] = [];
        }
        acc[curr.unit].push(curr);
        return acc;
    }, {});

    // Sort units if needed (Units are usually string sorted well enough: Unit 1, Unit 2...)
    const sortedUnits = Object.keys(groupedPrograms).sort();

    if (loading) return <div style={{ padding: '20px' }}>Loading exercises...</div>;

    return (
        <div className="examples-page" style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
            <h1>Programming Exercises</h1>
            <p>Practice with over 100+ defined examples for Python and Machine Learning.</p>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                <button
                    onClick={() => setActiveTab('Python')}
                    style={{
                        padding: '10px 20px',
                        cursor: 'pointer',
                        backgroundColor: activeTab === 'Python' ? '#007bff' : '#ddd',
                        color: activeTab === 'Python' ? 'white' : 'black',
                        border: 'none',
                        borderRadius: '5px',
                        fontSize: '1em'
                    }}
                >
                    Python Exercises
                </button>
                <button
                    onClick={() => setActiveTab('Machine Learning')}
                    style={{
                        padding: '10px 20px',
                        cursor: 'pointer',
                        backgroundColor: activeTab === 'Machine Learning' ? '#28a745' : '#ddd',
                        color: activeTab === 'Machine Learning' ? 'white' : 'black',
                        border: 'none',
                        borderRadius: '5px',
                        fontSize: '1em'
                    }}
                >
                    Machine Learning Exercises
                </button>
            </div>

            {/* Content */}
            {sortedUnits.map(unit => (
                <div key={unit} style={{ marginBottom: '30px', border: '1px solid #eee', borderRadius: '8px', overflow: 'hidden' }}>
                    <div style={{ backgroundColor: '#f9f9f9', padding: '15px', borderBottom: '1px solid #eee' }}>
                        <h3 style={{ margin: 0, color: '#333' }}>{unit}</h3>
                    </div>
                    <div style={{ padding: '0' }}>
                        {groupedPrograms[unit].map((prog, index) => (
                            <div key={prog.id} style={{
                                padding: '15px',
                                borderBottom: index !== groupedPrograms[unit].length - 1 ? '1px solid #f0f0f0' : 'none',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '10px'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div style={{ flex: 1 }}>
                                        <span style={{ fontWeight: 'bold', color: '#666', marginRight: '10px' }}>#{prog.id}</span>
                                        {prog.question}
                                    </div>
                                    <div style={{ marginLeft: '20px' }}>
                                        <Link to="/editor" state={{ question: prog.question, course: prog.course, code: prog.code }} style={{
                                            textDecoration: 'none',
                                            color: '#007bff',
                                            fontSize: '0.9em',
                                            border: '1px solid #007bff',
                                            padding: '5px 10px',
                                            borderRadius: '3px'
                                        }}>
                                            Try It
                                        </Link>
                                    </div>
                                </div>
                                <details style={{ marginTop: '5px' }}>
                                    <summary style={{ cursor: 'pointer', color: '#555', fontSize: '0.9em', marginBottom: '5px' }}>Show Solution</summary>
                                    <pre style={{
                                        backgroundColor: '#f4f4f4',
                                        padding: '10px',
                                        borderRadius: '5px',
                                        overflowX: 'auto',
                                        fontSize: '0.9em',
                                        border: '1px solid #ddd',
                                        margin: 0
                                    }}>
                                        <code>{prog.code || "# No solution provided"}</code>
                                    </pre>
                                </details>
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            {sortedUnits.length === 0 && (
                <p>No examples found for this category.</p>
            )}
        </div>
    );
};

export default ExamplesPage;
