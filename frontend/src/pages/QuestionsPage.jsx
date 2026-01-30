import React, { useState, useEffect } from 'react';

import API_URL from '../config';

const QuestionsPage = () => {
    const [questionsData, setQuestionsData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeSection, setActiveSection] = useState('Part A (1-Mark)');
    const [selectedUnit, setSelectedUnit] = useState('All');

    useEffect(() => {
        fetch(`${API_URL}/api/exam-questions`)
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

    return (
        <div className="questions-page" style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h1 style={{ marginBottom: '10px' }}>Exam Question Bank</h1>
            <p style={{ marginBottom: '20px', color: '#666' }}>Browse standard exam questions by subject and unit.</p>

            {/* Hierarchical Display */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {['Python', 'Machine Learning'].map(subject => {
                    // Filter Units for this Subject
                    const subjectUnits = questionsData.filter(u =>
                        subject === 'Python' ? u.unit.includes('Python') : !u.unit.includes('Python')
                    );

                    if (subjectUnits.length === 0) return null;

                    return (
                        <div key={subject} style={{ border: '1px solid #ddd', borderRadius: '8px', overflow: 'hidden' }}>
                            {/* Subject Header */}
                            <div style={{ backgroundColor: '#282a35', color: 'white', padding: '15px' }}>
                                <h2 style={{ margin: 0, fontSize: '1.5em' }}>{subject}</h2>
                            </div>

                            <div style={{ padding: '20px', backgroundColor: '#f9f9f9' }}>
                                {subjectUnits.map((unitData, uIdx) => (
                                    <details key={uIdx} style={{ marginBottom: '15px', backgroundColor: 'white', borderRadius: '5px', border: '1px solid #e0e0e0' }} open={false}>
                                        <summary style={{ padding: '15px', cursor: 'pointer', fontWeight: 'bold', listStyle: 'none', display: 'flex', alignItems: 'center' }}>
                                            <span style={{ marginRight: '10px', fontSize: '1.2em' }}>▶</span>
                                            {unitData.unit}
                                        </summary>

                                        <div style={{ padding: '0 20px 20px 20px' }}>
                                            {unitData.sections && Object.keys(unitData.sections).map((sectionTitle, sIdx) => (
                                                <details key={sIdx} style={{ marginBottom: '10px', paddingLeft: '20px' }} open={false}>
                                                    <summary style={{ padding: '10px', cursor: 'pointer', color: '#007bff', fontWeight: '500' }}>
                                                        <span style={{ marginRight: '5px' }}>▶</span> {sectionTitle}
                                                    </summary>

                                                    <div style={{ padding: '10px 0 10px 20px' }}>
                                                        {unitData.sections[sectionTitle].map((q, qIdx) => (
                                                            <div key={qIdx} style={{ marginBottom: '20px', borderLeft: '3px solid #eee', paddingLeft: '15px' }}>
                                                                <details>
                                                                    <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                                                                        Q{qIdx + 1}: {q.question}
                                                                    </summary>
                                                                    <div style={{
                                                                        marginTop: '10px',
                                                                        padding: '15px',
                                                                        backgroundColor: '#f8f9fa',
                                                                        borderLeft: '4px solid #28a745',
                                                                        borderRadius: '4px',
                                                                        whiteSpace: 'pre-wrap',
                                                                        fontFamily: '"Fira Code", monospace',
                                                                        lineHeight: '1.6'
                                                                    }}>
                                                                        {q.answer}
                                                                    </div>
                                                                </details>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </details>
                                            ))}
                                        </div>
                                    </details>
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>

            {questionsData.length === 0 && !loading && (
                <p>No questions found.</p>
            )}
        </div>
    );
};

export default QuestionsPage;
