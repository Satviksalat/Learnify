import React, { useState, useEffect } from 'react';

const QuizzesPage = () => {
    const [allQuizzes, setAllQuizzes] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('Python');
    const [selectedUnit, setSelectedUnit] = useState(null);
    const [userAnswers, setUserAnswers] = useState({});
    const [showResults, setShowResults] = useState(false);

    useEffect(() => {
        fetch('http://192.168.5.138:5004/api/quizzes')
            .then(res => res.json())
            .then(data => {
                setAllQuizzes(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error fetching quizzes:", err);
                setLoading(false);
            });
    }, []);

    const resetQuiz = () => {
        setSelectedUnit(null);
        setUserAnswers({});
        setShowResults(false);
    };

    // 1. Get unique units for the active tab (Course)
    const availableUnits = [...new Set(
        allQuizzes
            .filter(q => q.course === activeTab)
            .map(q => q.unit)
    )].sort();

    // 2. Start Quiz Handler
    const handleUnitSelect = (unit) => {
        setSelectedUnit(unit);
        setUserAnswers({});
        setShowResults(false);
    };

    // 3. Quiz Data for Selected Unit
    const activeQuestions = allQuizzes.filter(q => q.unit === selectedUnit);

    // 4. Handle Answer Selection
    const handleOptionSelect = (qId, option) => {
        setUserAnswers(prev => ({
            ...prev,
            [qId]: option
        }));
    };

    // 5. Calculate Score
    const calculateScore = () => {
        let score = 0;
        activeQuestions.forEach(q => {
            if (userAnswers[q.id] === q.correct_answer) {
                score++;
            }
        });
        return score;
    };

    if (loading) return <div style={{ padding: '20px' }}>Loading Quizzes...</div>;

    return (
        <div className="quizzes-page" style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>Interactive Quizzes</h1>

            {/* Navigation / Reset */}
            {selectedUnit && (
                <button
                    onClick={resetQuiz}
                    style={{
                        marginBottom: '20px',
                        padding: '5px 10px',
                        cursor: 'pointer',
                        background: '#eee',
                        border: '1px solid #ccc',
                        borderRadius: '4px'
                    }}
                >
                    &larr; Back to Topics
                </button>
            )}

            {!selectedUnit ? (
                // VIEW 1: Topic Selection
                <div>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                        <button
                            onClick={() => setActiveTab('Python')}
                            style={{
                                padding: '10px 20px',
                                cursor: 'pointer',
                                backgroundColor: activeTab === 'Python' ? '#007bff' : '#ddd',
                                color: activeTab === 'Python' ? 'white' : 'black',
                                border: 'none',
                                borderRadius: '5px'
                            }}
                        >
                            Python
                        </button>
                        <button
                            onClick={() => setActiveTab('Machine Learning')}
                            style={{
                                padding: '10px 20px',
                                cursor: 'pointer',
                                backgroundColor: activeTab === 'Machine Learning' ? '#28a745' : '#ddd',
                                color: activeTab === 'Machine Learning' ? 'white' : 'black',
                                border: 'none',
                                borderRadius: '5px'
                            }}
                        >
                            Machine Learning
                        </button>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
                        {availableUnits.length > 0 ? availableUnits.map(unit => (
                            <div
                                key={unit}
                                onClick={() => handleUnitSelect(unit)}
                                style={{
                                    padding: '20px',
                                    border: '1px solid #ddd',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    backgroundColor: '#f9f9f9',
                                    textAlign: 'center',
                                    transition: 'transform 0.2s',
                                    boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                                }}
                                onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                                onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                            >
                                <h3 style={{ margin: 0, fontSize: '1.1em', color: '#333' }}>{unit}</h3>
                                <p style={{ color: '#666', fontSize: '0.9em', marginTop: '5px' }}>
                                    Take Quiz &raquo;
                                </p>
                            </div>
                        )) : (
                            <p>No quizzes available for this course yet.</p>
                        )}
                    </div>
                </div>
            ) : (
                // VIEW 2: Active Quiz
                <div>
                    <h2 style={{ borderBottom: '2px solid #eee', paddingBottom: '10px' }}>{selectedUnit} Quiz</h2>

                    {activeQuestions.map((q, index) => {
                        const isCorrect = userAnswers[q.id] === q.correct_answer;
                        const userAnswer = userAnswers[q.id];

                        return (
                            <div key={q.id} style={{ marginBottom: '30px', padding: '15px', border: '1px solid #eee', borderRadius: '8px' }}>
                                <p style={{ fontWeight: 'bold', fontSize: '1.1em' }}>
                                    {index + 1}. {q.question}
                                </p>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                    {q.options.map(option => {
                                        let bgColor = '#f9f9f9';
                                        let borderColor = '#ddd';

                                        if (showResults) {
                                            if (option === q.correct_answer) {
                                                bgColor = '#d4edda'; // Green for correct
                                                borderColor = '#c3e6cb';
                                            } else if (option === userAnswer && option !== q.correct_answer) {
                                                bgColor = '#f8d7da'; // Red for wrong selection
                                                borderColor = '#f5c6cb';
                                            }
                                        } else if (userAnswer === option) {
                                            bgColor = '#e7f1ff'; // Selected state
                                            borderColor = '#b8daff';
                                        }

                                        return (
                                            <label
                                                key={option}
                                                style={{
                                                    padding: '10px',
                                                    border: `1px solid ${borderColor}`,
                                                    borderRadius: '5px',
                                                    backgroundColor: bgColor,
                                                    cursor: showResults ? 'default' : 'pointer',
                                                    display: 'block'
                                                }}
                                            >
                                                <input
                                                    type="radio"
                                                    name={`q-${q.id}`}
                                                    value={option}
                                                    checked={userAnswer === option}
                                                    onChange={() => !showResults && handleOptionSelect(q.id, option)}
                                                    disabled={showResults}
                                                    style={{ marginRight: '10px' }}
                                                />
                                                {option}
                                            </label>
                                        );
                                    })}
                                </div>
                                {showResults && (
                                    <div style={{ marginTop: '10px', fontSize: '0.9em', color: isCorrect ? 'green' : 'red' }}>
                                        {isCorrect ? "✅ Correct!" : `❌ Incorrect. The correct answer is: ${q.correct_answer}`}
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {!showResults ? (
                        <button
                            onClick={() => setShowResults(true)}
                            disabled={Object.keys(userAnswers).length !== activeQuestions.length}
                            style={{
                                padding: '15px 30px',
                                backgroundColor: Object.keys(userAnswers).length === activeQuestions.length ? '#28a745' : '#ccc',
                                color: 'white',
                                border: 'none',
                                borderRadius: '5px',
                                fontSize: '1.2em',
                                cursor: Object.keys(userAnswers).length === activeQuestions.length ? 'pointer' : 'not-allowed',
                                width: '100%'
                            }}
                        >
                            Submit Quiz
                        </button>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '10px' }}>
                            <h2>Your Score: {calculateScore()} / {activeQuestions.length}</h2>
                            <p>{calculateScore() === activeQuestions.length ? "Perfect Score! 🎉" : "Keep practicing! 💪"}</p>
                            <button
                                onClick={resetQuiz}
                                style={{
                                    marginTop: '10px',
                                    padding: '10px 20px',
                                    backgroundColor: '#007bff',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Take Another Quiz
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default QuizzesPage;
