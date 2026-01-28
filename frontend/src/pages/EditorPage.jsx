import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

import ProModal from '../components/ProModal';

const EditorPage = () => {
    const location = useLocation();
    const [code, setCode] = useState("");
    const [srcDoc, setSrcDoc] = useState("");
    const [tech, setTech] = useState("html");
    const [showPro, setShowPro] = useState(false);

    useEffect(() => {
        if (location.state) {
            if (location.state.code) {
                setCode(location.state.code);
                setSrcDoc(location.state.code);
                if (location.state.course) {
                    const isML = location.state.course === "Machine Learning";
                    setTech(isML ? 'ml_python' : 'python');
                }
            } else if (location.state.question) {
                // Pre-fill from Examples page
                const q = location.state.question;
                const isML = location.state.course === "Machine Learning";
                const starter = isML
                    ? `'''\n${q}\n'''\nimport pandas as pd\nimport numpy as np\n\n# Write your code here\nprint("ML Task Loaded")`
                    : `'''\n${q}\n'''\n\n# Write your code here\nprint("Hello World")`;

                setCode(starter);
                setTech(isML ? 'ml_python' : 'python');
                setSrcDoc(starter);
            }

            if (location.state.technology) {
                let t = location.state.technology;
                if (t === "Python Programming Course") t = "python";
                if (t === "Machine Learning with Python") t = "ml_python";
                setTech(t);
            }
        } else {
            // Default code if none provided
            const defaultCode = `<!DOCTYPE html>
<html>
<body>

<h1>Hello World</h1>
<p>This is a paragraph.</p>

</body>
</html>`;
            setCode(defaultCode);
            setSrcDoc(defaultCode);
            setTech("html");
        }
    }, [location]);

    const handleRun = async () => {
        if (['python', 'ml_python'].includes(tech)) {
            setSrcDoc(`<html><body>Running...</body></html>`);
            try {
                const response = await fetch('https://learnify-api-ohc0.onrender.com/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, language: tech })
                });
                const data = await response.json();

                const escapeHtml = (text) => {
                    if (!text) return "";
                    return text
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#039;");
                };

                let outputHtml = `
                    <html>
                    <body style="background:#282A35; color: white; font-family: monospace; white-space: pre-wrap; padding: 20px;">
                        <div>${escapeHtml(data.output)}</div>
                `;

                if (data.image) {
                    outputHtml += `<div style="margin-top:20px; border-top:1px solid #444; padding-top:20px;"><img src="data:image/png;base64,${data.image}" style="max-width:100%"/></div>`;
                }

                outputHtml += `</body></html>`;
                setSrcDoc(outputHtml);

            } catch (err) {
                setSrcDoc(`<html><body>Error: ${err.message}</body></html>`);
            }
        } else {
            setSrcDoc(code);
        }
    };

    return (
        <div className="editor-container">
            {showPro && <ProModal onClose={() => setShowPro(false)} />}
            <div className="editor-pane">
                <div className="editor-header">
                    <span>Source Code ({tech})</span>
                    <div>
                        <button
                            className="run-btn"
                            style={{ marginRight: '10px', backgroundColor: '#f44336' }}
                            onClick={() => setShowPro(true)}
                        >
                            💾 Save
                        </button>
                        <button className="run-btn" onClick={handleRun}>Run &raquo;</button>
                    </div>
                </div>
                <textarea
                    className="code-input"
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    spellCheck="false"
                />
            </div>
            <div className="preview-pane">
                <div className="editor-header">
                    <span>Result</span>
                </div>
                <iframe
                    className="preview-frame"
                    srcDoc={srcDoc}
                    title="output"
                    sandbox="allow-scripts allow-same-origin"
                />
            </div>
        </div>
    );
};

export default EditorPage;
