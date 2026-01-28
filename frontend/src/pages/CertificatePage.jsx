import React, { useEffect, useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { getCompletedTutorials } from '../utils/progress';

const CertificatePage = () => {
    const location = useLocation();
    const tech = new URLSearchParams(location.search).get('tech') || 'Python Programming Course';
    const [name, setName] = useState("");
    const [progress, setProgress] = useState(0);
    const [total, setTotal] = useState(0);
    const [completed, setCompleted] = useState(0);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('https://learnify-api-ohc0.onrender.com/api/tutorials')
            .then(res => res.json())
            .then(data => {
                // Filter tutorials for this specific technology
                const courseTutorials = data.filter(t => t.technology === tech);
                const totalCount = courseTutorials.length;

                // Get completed IDs
                const completedIds = getCompletedTutorials();

                // Count how many of THIS course's tutorials are completed
                const completedCount = courseTutorials.filter(t => completedIds.includes(t.id)).length;

                setTotal(totalCount);
                setCompleted(completedCount);
                setProgress(totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, [tech]);

    if (loading) return <div style={{ padding: '20px' }}>Checking eligibility...</div>;

    // View 1: Incomplete
    if (progress < 100) {
        return (
            <div style={{ padding: '50px', textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
                <h1>🎓 Certificate Eligibility</h1>
                <p style={{ fontSize: '1.2em', color: '#555' }}>
                    You have completed <strong>{completed}</strong> out of <strong>{total}</strong> tutorials for the {tech}.
                </p>

                {/* Progress Bar */}
                <div style={{
                    width: '100%',
                    maxWidth: '500px',
                    height: '30px',
                    backgroundColor: '#e0e0e0',
                    borderRadius: '15px',
                    margin: '30px auto',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        width: `${progress}%`,
                        height: '100%',
                        backgroundColor: '#4CAF50',
                        lineHeight: '30px',
                        color: 'white',
                        fontWeight: 'bold',
                        transition: 'width 0.5s ease-in-out'
                    }}>
                        {progress}%
                    </div>
                </div>

                <div style={{ marginTop: '40px' }}>
                    <p>Keep learning! Complete all topics to unlock your certificate.</p>
                    <Link to="/" style={{
                        display: 'inline-block',
                        textDecoration: 'none',
                        backgroundColor: '#007bff',
                        color: 'white',
                        padding: '10px 20px',
                        borderRadius: '5px'
                    }}>
                        Back to Course
                    </Link>
                </div>
            </div>
        );
    }

    // View 2: Certificate
    return (
        <div style={{ padding: '50px', textAlign: 'center', backgroundColor: '#f9f9f9', minHeight: '100vh', fontFamily: 'Arial, sans-serif' }}>
            <div className="no-print" style={{ marginBottom: '20px' }}>
                <h1>🎉 Congratulations!</h1>
                <p>You have successfully completed the course.</p>
                <p>Enter your name below to personalize your certificate.</p>
            </div>

            <div className="certificate-container" style={{
                border: '10px solid #787878',
                padding: '50px',
                backgroundColor: 'white',
                maxWidth: '800px',
                margin: '0 auto',
                boxShadow: '0 0 20px rgba(0,0,0,0.1)',
                position: 'relative'
            }}>
                {/* Seal Mockup */}
                <div style={{
                    position: 'absolute',
                    bottom: '50px',
                    right: '50px',
                    width: '100px',
                    height: '100px',
                    backgroundColor: '#FFD700',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 10px rgba(0,0,0,0.3)',
                    color: '#b8860b',
                    fontWeight: 'bold',
                    border: '4px solid #fff'
                }}>
                    SEAL
                </div>

                <div style={{ fontSize: '40px', fontWeight: 'bold', color: '#282a35', fontFamily: 'Garamond, serif' }}>Certificate of Completion</div>

                <div style={{ fontSize: '20px', margin: '30px 0' }}>This is to certify that</div>

                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="[Enter Your Name]"
                    style={{
                        fontSize: '36px',
                        border: 'none',
                        borderBottom: '2px solid #ccc',
                        textAlign: 'center',
                        width: '80%',
                        fontFamily: "'Segoe Script', cursive",
                        color: '#333',
                        background: 'transparent',
                        outline: 'none'
                    }}
                />

                <div style={{ fontSize: '20px', margin: '30px 0' }}>has demonstrated mastery by completing the</div>

                <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#4CAF50', textTransform: 'uppercase', margin: '20px 0' }}>
                    {tech}
                </div>

                <div style={{ fontSize: '20px', margin: '20px 0' }}>on the Interactive Tutorial Platform</div>

                <div style={{ marginTop: '60px', display: 'flex', justifyContent: 'space-between', padding: '0 50px' }}>
                    <div style={{ borderTop: '1px solid #333', width: '200px', paddingTop: '10px' }}>
                        {new Date().toLocaleDateString()}
                        <div style={{ fontSize: '12px', color: '#888' }}>Date</div>
                    </div>
                    <div style={{ borderTop: '1px solid #333', width: '200px', paddingTop: '10px' }}>
                        Learnify System
                        <div style={{ fontSize: '12px', color: '#888' }}>Instructor</div>
                    </div>
                </div>
            </div>

            <div className="no-print" style={{ marginTop: '30px' }}>
                <button
                    onClick={() => window.print()}
                    style={{
                        padding: '15px 30px',
                        fontSize: '18px',
                        backgroundColor: '#282a35',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        borderRadius: '5px',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                    }}
                >
                    🖨️ Print / Save as PDF
                </button>
            </div>

            <style>{`
                @media print {
                    .no-print {
                        display: none;
                    }
                    body {
                        background-color: white;
                    }
                    .certificate-container {
                        box-shadow: none !important;
                        margin: 0 !important;
                        width: 100%;
                    }
                }
            `}</style>
        </div>
    );
};

export default CertificatePage;
