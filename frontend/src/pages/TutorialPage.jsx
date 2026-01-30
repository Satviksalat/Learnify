import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import AdBanner from '../components/AdBanner';
import { isTutorialCompleted, toggleTutorialCompletion } from '../utils/progress';

import API_URL from '../config';

const TutorialPage = () => {
    const { id } = useParams();
    const [tutorial, setTutorial] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isCompleted, setIsCompleted] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetch(`${API_URL}/api/tutorial/${id}`)
            .then(res => res.json())
            .then(data => {
                setTutorial(data);
                setIsCompleted(isTutorialCompleted(id));

                // Save As Last Visited
                localStorage.setItem('lastTutorial', JSON.stringify({
                    id: id,
                    title: data.title,
                    technology: data.technology
                }));

                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, [id]);

    const handleCompletion = () => {
        const newState = toggleTutorialCompletion(id);
        setIsCompleted(newState);
    };

    if (loading) return <div className="main-content">Loading...</div>;
    if (!tutorial) return <div className="main-content">Tutorial not found</div>;

    return (
        <div className="main-content">
            <h1 className="tutorial-title">{tutorial.title}</h1>
            <p className="tutorial-definition">{tutorial.description || tutorial.definition}</p>
            <hr />

            <AdBanner slot="Top" />

            {tutorial.explanation && (
                <div dangerouslySetInnerHTML={{ __html: tutorial.explanation }} />
            )}

            {tutorial.code_example && (
                <div className="example-section">
                    <h3>Example</h3>
                    <div className="code-box">
                        <pre>{tutorial.code_example}</pre>
                    </div>
                    {/* Always show button if code exists, or use data flag */}
                    <Link
                        to="/editor"
                        state={{ code: tutorial.code_example, technology: tutorial.technology }}
                        className="try-it-btn"
                    >
                        Try It Yourself &raquo;
                    </Link>
                </div>
            )}

            <AdBanner slot="Middle" />

            {tutorial.key_points && (
                <div>
                    <h3>Key Points</h3>
                    <ul>
                        {tutorial.key_points.map((point, index) => (
                            <li key={index}>{point}</li>
                        ))}
                    </ul>
                </div>
            )}

            <div style={{ marginTop: '40px', textAlign: 'center' }}>
                <button
                    onClick={handleCompletion}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: isCompleted ? '#28a745' : '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer',
                        marginRight: '10px',
                        opacity: isCompleted ? 0.9 : 1
                    }}
                >
                    {isCompleted ? "✅ Marked as Complete" : "Mark as Complete"}
                </button>
                <Link
                    to={`/certificate?tech=${tutorial.technology}`}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#2196F3',
                        color: 'white',
                        textDecoration: 'none',
                        borderRadius: '5px'
                    }}
                >
                    Get Certificate
                </Link>
            </div>
        </div>
    );
};

export default TutorialPage;
