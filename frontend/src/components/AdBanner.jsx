import React from 'react';

const AdBanner = ({ slot }) => {
    return (
        <div className="ad-banner" style={{
            margin: '20px 0',
            padding: '20px',
            backgroundColor: '#eee',
            border: '1px solid #ddd',
            textAlign: 'center',
            color: '#555',
            fontSize: '0.9rem'
        }}>
            <small>Advertisement ({slot})</small>
            <div style={{ fontWeight: 'bold', marginTop: '5px' }}>
                Start Your Web Dev Journey Today!
            </div>
            <button style={{
                marginTop: '10px',
                padding: '5px 10px',
                backgroundColor: '#282a35',
                color: 'white',
                border: 'none',
                cursor: 'pointer'
            }}>
                Learn More
            </button>
        </div>
    );
};

export default AdBanner;
