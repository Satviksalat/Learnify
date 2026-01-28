import React, { useState, useEffect } from 'react';

const QuestionsPage = () => {
    const [questionsData, setQuestionsData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeSection, setActiveSection] = useState('Part A (1-Mark)');
    const [selectedUnit, setSelectedUnit] = useState('All');

    useEffect(() => {
        fetch('https://learnify-api-ohc0.onrender.com/api/exam-questions')
            .then(res => res.json())
            .then(data => {
                setQuestionsData(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error fetching exam questions:", err);
                setLoading(false);
            });
    }, []);

    // Get unique units
    const units = ['All', ...new Set(questionsData.map(item => item.unit))];

    // Filter content
    const getFilteredContent = () => {
        let filtered = questionsData;
        if (selectedUnit !== 'All') {
            filtered = filtered.filter(item => item.unit === selectedUnit);
        }

        // Aggregate questions from the filtered units for the active section
        let aggregatedQuestions = [];
        filtered.forEach(unitData => {
            if (unitData.sections && unitData.sections[activeSection]) {
                const questionsWithUnit = unitData.sections[activeSection].map(q => ({
                    ...q,
                    unitName: unitData.unit
                }));
                aggregatedQuestions = aggregatedQuestions.concat(questionsWithUnit);
            }
        });
        return aggregatedQuestions;
    };

    const questionsToShow = getFilteredContent();

    if (loading) return <div style={{ padding: '20px' }}>Loading Question Bank...</div>;

    const sections = ['Part A (1-Mark)', 'Part B (2-Mark)', 'Part C (3-Mark)', 'Part D (5-Mark)'];

    return (
        <div className="questions-page" style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h1 style={{ marginBottom: '10px' }}>Exam Question Bank</h1>
            <p style={{ marginBottom: '20px', color: '#666' }}>Browse standard exam questions by unit and marks.</p>

            {/* Controls Container */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginBottom: '30px' }}>

                {/* Unit Filter */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontWeight: 'bold' }}>Filter by Unit:</span>
                    <select
                        value={selectedUnit}
                        onChange={(e) => setSelectedUnit(e.target.value)}
                        style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                    >
                        {units.map(u => <option key={u} value={u}>{u}</option>)}
                    </select>
                </div>

                {/* Section Tabs */}
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    {sections.map(section => (
                        <button
                            key={section}
                            onClick={() => setActiveSection(section)}
                            style={{
                                padding: '10px 20px',
                                cursor: 'pointer',
                                backgroundColor: activeSection === section ? '#007bff' : '#f0f0f0',
                                color: activeSection === section ? 'white' : '#333',
                                border: 'none',
                                borderRadius: '5px',
                                fontWeight: '500',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }}
                        >
                            {section}
                        </button>
                    ))}
                </div>
            </div>

            {/* Questions Grid */}
            <div style={{ display: 'grid', gap: '20px' }}>
                {questionsToShow.length === 0 ? (
                    <p>No questions found for this selection.</p>
                ) : (
                    questionsToShow.map((q, idx) => (
                        <div key={idx} style={{
                            border: '1px solid #e0e0e0',
                            borderRadius: '8px',
                            padding: '20px',
                            backgroundColor: '#fff',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                        }}>
                            {/* Meta info */}
                            <div style={{
                                fontSize: '0.85em',
                                color: '#888',
                                marginBottom: '10px',
                                textTransform: 'uppercase',
                                letterSpacing: '0.5px'
                            }}>
                                {q.unitName} • {activeSection}
                            </div>

                            {/* Question */}
                            <h3 style={{
                                marginTop: '0',
                                marginBottom: '15px',
                                color: '#333',
                                fontSize: '1.2rem'
                            }}>
                                Q: {q.question}
                            </h3>

                            {/* Answer - STRICT FORMATTING */}
                            <div style={{
                                backgroundColor: '#f8f9fa',
                                padding: '15px',
                                borderRadius: '6px',
                                borderLeft: '4px solid #28a745',
                                fontFamily: '"Fira Code", "Courier New", monospace', // Monospace for alignment
                                whiteSpace: 'pre-wrap', // Preserves newlines and spacing
                                color: '#212529',
                                lineHeight: '1.6',
                                fontSize: '0.95rem'
                            }}>
                                {q.answer}
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default QuestionsPage;
