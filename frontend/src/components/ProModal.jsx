import React from 'react';

const ProModal = ({ onClose }) => {
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.7)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000
        }}>
            <div style={{
                backgroundColor: 'white',
                padding: '40px',
                borderRadius: '10px',
                textAlign: 'center',
                maxWidth: '500px',
                position: 'relative'
            }}>
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px',
                        border: 'none',
                        background: 'none',
                        fontSize: '20px',
                        cursor: 'pointer'
                    }}
                >
                    &times;
                </button>

                <h2 style={{ color: '#282a35' }}>Upgrade to PRO</h2>
                <div style={{ fontSize: '50px', margin: '20px 0' }}>👑</div>
                <p>Unlock advanced features to accelerate your learning:</p>

                <ul style={{ textAlign: 'left', margin: '20px auto', display: 'inline-block' }}>
                    <li>💾 Save your code snippets showing you</li>
                    <li>🌙 Premium Dark Themes</li>
                    <li>🚫 Ad-free Experience</li>
                    <li>📜 Verified Certificates</li>
                </ul>

                <button style={{
                    display: 'block',
                    width: '100%',
                    padding: '15px',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    fontSize: '18px',
                    cursor: 'pointer',
                    marginTop: '20px'
                }}>
                    Get PRO for $9.99/mo
                </button>
            </div>
        </div>
    );
};

export default ProModal;
