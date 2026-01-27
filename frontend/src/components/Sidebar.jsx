import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import AdBanner from './AdBanner';
import { getCompletedTutorials } from '../utils/progress';

const Sidebar = ({ isOpen, closeSidebar }) => {
    const [tutorials, setTutorials] = useState([]);
    const [completedIds, setCompletedIds] = useState([]);
    const location = useLocation();

    useEffect(() => {
        // Fetch Data - Using Network IP
        fetch('http://192.168.5.138:5004/api/tutorials')
            .then(res => res.json())
            .then(data => setTutorials(data))
            .catch(err => console.error("Failed to fetch tutorials", err));

        // Initial Progress Load
        setCompletedIds(getCompletedTutorials());

        // Listen for updates
        const handleProgressUpdate = () => {
            setCompletedIds(getCompletedTutorials());
        };
        window.addEventListener('progressUpdated', handleProgressUpdate);

        return () => window.removeEventListener('progressUpdated', handleProgressUpdate);
    }, []);

    const [expanded, setExpanded] = useState({});

    // Group tutorials: Technology -> Unit -> List
    const grouped = tutorials.reduce((acc, t) => {
        const tech = t.technology || 'Other';
        const unit = t.unit || 'General';

        if (!acc[tech]) acc[tech] = {};
        if (!acc[tech][unit]) acc[tech][unit] = [];

        acc[tech][unit].push(t);
        return acc;
    }, {});

    // Initialize state when data loads or route changes
    useEffect(() => {
        if (Object.keys(grouped).length > 0) {
            let nextExpanded = {};

            // 1. Try to load from LocalStorage
            const savedState = localStorage.getItem('sidebarExpanded');
            if (savedState) {
                try {
                    nextExpanded = JSON.parse(savedState);
                } catch (e) {
                    console.error("Error parsing saved sidebar state", e);
                }
            }

            // 2. If no saved state, apply defaults (Technologies Open, Units Closed)
            if (Object.keys(nextExpanded).length === 0) {
                Object.keys(grouped).forEach(tech => {
                    nextExpanded[tech] = true;
                });
            }

            // 3. Auto-expand based on Active URL
            // URL format: /tutorial/:id
            if (location.pathname.startsWith('/tutorial/')) {
                const currentId = location.pathname.split('/tutorial/')[1];
                const currentTutorial = tutorials.find(t => t.id === currentId);

                if (currentTutorial) {
                    const tech = currentTutorial.technology || 'Other';
                    const unit = currentTutorial.unit || 'General';

                    // Respect user preference: Only auto-expand if state is unknown (undefined).
                    // If user collapsed it (false) or expanded it (true) in the past, keep that state.
                    if (nextExpanded[tech] === undefined) {
                        nextExpanded[tech] = true;
                    }
                    const unitKey = `${tech}-${unit}`;
                    if (nextExpanded[unitKey] === undefined) {
                        nextExpanded[unitKey] = true;
                    }
                }
            }

            // Update state
            setExpanded(prev => ({ ...prev, ...nextExpanded }));
        }
    }, [tutorials, location.pathname]); // Re-run if tutorials load or URL changes

    const toggle = (key) => {
        setExpanded(prev => {
            const newState = { ...prev, [key]: !prev[key] };
            localStorage.setItem('sidebarExpanded', JSON.stringify(newState));
            return newState;
        });
    };

    return (
        <>
            {/* Mobile Overlay */}
            <div
                className={`sidebar-overlay ${isOpen ? 'open' : ''}`}
                onClick={closeSidebar}
            />

            <div className={`sidebar ${isOpen ? 'open' : ''}`}>
                {/* Close Button for Mobile */}
                <button className="close-sidebar-btn" onClick={closeSidebar}>&times;</button>

                {Object.keys(grouped).length === 0 ? <p style={{ padding: '10px', color: '#888' }}>Loading...</p> : null}

                {Object.keys(grouped).map(tech => (
                    <div key={tech} className="sidebar-group">
                        <h3 onClick={() => toggle(tech)} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#eee', padding: '8px', margin: '5px 0' }}>
                            {tech}
                            <span>{expanded[tech] ? '−' : '+'}</span>
                        </h3>

                        {expanded[tech] && (
                            <div className="sidebar-tech-content" style={{ paddingLeft: '10px' }}>
                                {Object.keys(grouped[tech]).map(unit => (
                                    <div key={unit} className="sidebar-unit">
                                        <h4 onClick={() => toggle(`${tech}-${unit}`)} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '5px 0 5px 0', color: '#555' }}>
                                            {unit}
                                            <span style={{ fontSize: '0.8em' }}>{expanded[`${tech}-${unit}`] ? '▼' : '▶'}</span>
                                        </h4>

                                        {expanded[`${tech}-${unit}`] && (
                                            <div className="sidebar-links" style={{ paddingLeft: '10px', display: 'flex', flexDirection: 'column' }}>
                                                {grouped[tech][unit].map(t => (
                                                    <Link
                                                        key={t.id}
                                                        to={`/tutorial/${t.id}`}
                                                        style={{
                                                            textDecoration: 'none',
                                                            color: '#007bff',
                                                            padding: '2px 0',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'space-between'
                                                        }}
                                                        onClick={closeSidebar} // Close sidebar on link click (Mobile UX)
                                                    >
                                                        <span>{t.title}</span>
                                                        {completedIds.includes(t.id) && <span style={{ color: 'green', marginLeft: '5px' }}>✅</span>}
                                                    </Link>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}

                {/* Question Bank Link */}
                <div className="sidebar-group" style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '10px' }}>
                    <Link to="/questions" onClick={closeSidebar} style={{
                        display: 'block',
                        padding: '10px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        textDecoration: 'none',
                        borderRadius: '4px',
                        textAlign: 'center',
                        fontWeight: 'bold'
                    }}>
                        📚 Question Bank
                    </Link>
                </div>

                <div className="sidebar-ad">
                    <AdBanner slot="Sidebar" />
                </div>
            </div>
        </>
    );
};

export default Sidebar;
