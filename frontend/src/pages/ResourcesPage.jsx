import React, { useState } from 'react';

const ResourcesPage = () => {
    const [activeTab, setActiveTab] = useState('python');
    const [examQuestions, setExamQuestions] = useState([]);
    const [expandedAns, setExpandedAns] = useState({});
    const [searchTerm, setSearchTerm] = useState('');

    // Fetch Exam Questions when tab is active
    React.useEffect(() => {
        if (activeTab === 'questions') {
            const apiUrl = import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com';
            fetch(`${apiUrl}/api/exam-questions`)
                .then(res => res.json())
                .then(data => setExamQuestions(data))
                .catch(err => console.error(err));
        }
    }, [activeTab]);

    const toggleAnswer = (unitIndex, type, qIndex) => {
        const key = `${unitIndex}-${type}-${qIndex}`;
        setExpandedAns(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const glossary = [
        { term: 'Algorithm', def: 'A step-by-step process to solve a problem.' },
        { term: 'Argument', def: 'A value passed to a function when it is called.' },
        { term: 'Boolean', def: 'A data type that has one of two possible values: True or False.' },
        { term: 'Class', def: 'A code template for creating objects.' },
        { term: 'Compiler', def: 'Translates code into machine language before execution.' },
        { term: 'Dictionary', def: 'A collection of key-value pairs.' },
        { term: 'Function', def: 'A block of code which only runs when it is called.' },
        { term: 'Immutable', def: 'An object whose state cannot be modified after it is created.' },
        { term: 'Index', def: 'The position of an item in a list or string (starts at 0).' },
        { term: 'Inheritance', def: 'Allows a class to derive features from another class.' },
        { term: 'Interpreter', def: 'Translates code line-by-line during execution.' },
        { term: 'List', def: 'A mutable, ordered sequence of elements.' },
        { term: 'Loop', def: 'Repeats a block of code multiple times.' },
        { term: 'Machine Learning', def: 'Algorithms that allow computers to learn from data.' },
        { term: 'Method', def: 'A function that belongs to an object.' },
        { term: 'Module', def: 'A file containing Python definitions and statements.' },
        { term: 'Mutable', def: 'An object whose state can be modified after creation.' },
        { term: 'Neural Network', def: 'A series of algorithms that mimic the human brain.' },
        { term: 'Object', def: 'An instance of a class.' },
        { term: 'Parameter', def: 'A variable listed inside the parentheses in the function definition.' },
        { term: 'Recursion', def: 'A function calling itself.' },
        { term: 'String', def: 'A sequence of characters.' },
        { term: 'Supervised Learning', def: 'Learning from labeled training data.' },
        { term: 'Syntax', def: 'The set of rules that defines the combinations of symbols.' },
        { term: 'Tuple', def: 'An immutable, ordered sequence of elements.' },
        { term: 'Unsupervised Learning', def: 'Learning from unlabeled data (finding patterns).' },
        { term: 'Variable', def: 'A container for storing data values.' },
    ];

    const filteredGlossary = glossary.filter(item =>
        item.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.def.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto', fontFamily: 'Arial, sans-serif' }}>
            <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>Student Resources</h1>

            {/* Navigation Tabs */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '30px', flexWrap: 'wrap' }}>
                {['python', 'ml', 'questions', 'glossary', 'downloads'].map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        style={{
                            padding: '10px 20px',
                            cursor: 'pointer',
                            backgroundColor: activeTab === tab ? '#282a35' : '#ddd',
                            color: activeTab === tab ? 'white' : '#333',
                            border: 'none',
                            borderRadius: '5px',
                            fontSize: '1em',
                            textTransform: 'capitalize'
                        }}
                    >
                        {tab === 'ml' ? 'ML Cheat Sheet' : tab === 'questions' ? 'Question Bank' : tab + (tab === 'python' ? ' Cheat Sheet' : '')}
                    </button>
                ))}
            </div>

            {/* Questions Bank Tab */}
            {activeTab === 'questions' && (
                <div className="question-bank">
                    <h2>Exam Question Bank (Structured Format)</h2>
                    <p style={{ marginBottom: '20px', color: '#666' }}>Prepare for your written exams with these structured practice questions.</p>

                    {examQuestions.map((unitData, uIdx) => (
                        <div key={uIdx} style={{ marginBottom: '40px', border: '1px solid #ddd', borderRadius: '8px', overflow: 'hidden' }}>
                            <div style={{ backgroundColor: '#282a35', color: 'white', padding: '15px', borderBottom: '1px solid #ddd' }}>
                                <h3 style={{ margin: 0 }}>{unitData.unit}</h3>
                            </div>

                            <div style={{ padding: '20px' }}>
                                {unitData.sections && Object.keys(unitData.sections).map((sectionTitle, sIdx) => (
                                    <div key={sIdx} style={{ marginBottom: '30px' }}>
                                        <h4 style={{
                                            borderBottom: '2px solid #007bff',
                                            display: 'inline-block',
                                            paddingBottom: '5px',
                                            marginBottom: '15px',
                                            color: '#007bff'
                                        }}>
                                            {sectionTitle}
                                        </h4>

                                        {unitData.sections[sectionTitle].map((q, qIdx) => (
                                            <div key={qIdx} style={{ marginBottom: '15px', paddingLeft: '15px', borderLeft: '3px solid #eee' }}>
                                                <p style={{ fontWeight: 'bold', marginBottom: '8px' }}>
                                                    {qIdx + 1}. {q.question}
                                                </p>

                                                <button
                                                    onClick={() => toggleAnswer(uIdx, sectionTitle, qIdx)}
                                                    style={{
                                                        background: 'none',
                                                        border: '1px solid #007bff',
                                                        color: '#007bff',
                                                        cursor: 'pointer',
                                                        padding: '5px 10px',
                                                        fontSize: '0.9em',
                                                        borderRadius: '4px'
                                                    }}
                                                >
                                                    {expandedAns[`${uIdx}-${sectionTitle}-${qIdx}`] ? 'Hide Answer' : 'Show Answer'}
                                                </button>

                                                {expandedAns[`${uIdx}-${sectionTitle}-${qIdx}`] && (
                                                    <div style={{ marginTop: '10px', color: '#333', backgroundColor: '#f4f4f4', padding: '15px', borderRadius: '5px', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                                                        <strong>Answer:</strong><br />
                                                        {q.answer}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}

                    {examQuestions.length === 0 && <p>Loading questions...</p>}
                </div>
            )}

            {/* Python Cheat Sheet */}
            {activeTab === 'python' && (
                <div className="cheat-sheet">
                    <h2>Python Quick Reference</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
                        <div className="card" style={cardStyle}>
                            <h3>Key Data Types</h3>
                            <table style={tableStyle}>
                                <tbody>
                                    <tr><td>int</td><td>Integer (1, -5)</td></tr>
                                    <tr><td>float</td><td>Decimal (1.5, 3.14)</td></tr>
                                    <tr><td>str</td><td>String ("Hello")</td></tr>
                                    <tr><td>bool</td><td>Boolean (True, False)</td></tr>
                                    <tr><td>list</td><td>[1, 2, 'a'] (Mutable)</td></tr>
                                    <tr><td>tuple</td><td>(1, 2) (Immutable)</td></tr>
                                    <tr><td>dict</td><td>{`{'a': 1}`} (Key-Value)</td></tr>
                                </tbody>
                            </table>
                        </div>

                        <div className="card" style={cardStyle}>
                            <h3>String Methods</h3>
                            <table style={tableStyle}>
                                <tbody>
                                    <tr><td>s.lower()</td><td>Lowercase</td></tr>
                                    <tr><td>s.upper()</td><td>Uppercase</td></tr>
                                    <tr><td>s.strip()</td><td>Remove whitespace</td></tr>
                                    <tr><td>s.replace(a, b)</td><td>Replace a with b</td></tr>
                                    <tr><td>s.split(sep)</td><td>Split string into list</td></tr>
                                    <tr><td>len(s)</td><td>Length of string</td></tr>
                                </tbody>
                            </table>
                        </div>

                        <div className="card" style={cardStyle}>
                            <h3>List Operations</h3>
                            <table style={tableStyle}>
                                <tbody>
                                    <tr><td>lst.append(x)</td><td>Add x to end</td></tr>
                                    <tr><td>lst.pop()</td><td>Remove last item</td></tr>
                                    <tr><td>lst.sort()</td><td>Sort list</td></tr>
                                    <tr><td>lst[i]</td><td>Access element at i</td></tr>
                                    <tr><td>x in lst</td><td>Check membership</td></tr>
                                    <tr><td>len(lst)</td><td>Count items</td></tr>
                                </tbody>
                            </table>
                        </div>

                        <div className="card" style={cardStyle}>
                            <h3>Control Flow</h3>
                            <pre style={{ backgroundColor: '#f4f4f4', padding: '10px', borderRadius: '5px', fontSize: '0.9em' }}>
                                {`if condition:
    # code
elif other:
    # code
else:
    # code

for i in range(5):
    print(i)

while x < 10:
    x += 1`}
                            </pre>
                        </div>
                    </div>
                </div>
            )}

            {/* ML Cheat Sheet */}
            {activeTab === 'ml' && (
                <div className="cheat-sheet">
                    <h2>Machine Learning Quick Reference</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
                        <div className="card" style={cardStyle}>
                            <h3>Type of Learning</h3>
                            <ul style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
                                <li><strong>Supervised:</strong> Input data is labeled (e.g., Regression, Classification).</li>
                                <li><strong>Unsupervised:</strong> Input data is unlabeled (e.g., Clustering).</li>
                                <li><strong>Reinforcement:</strong> Learning via rewards/punishment.</li>
                            </ul>
                        </div>

                        <div className="card" style={cardStyle}>
                            <h3>Common Libraries</h3>
                            <table style={tableStyle}>
                                <tbody>
                                    <tr><td>NumPy</td><td>Math & Arrays</td></tr>
                                    <tr><td>Pandas</td><td>Data DataFrames</td></tr>
                                    <tr><td>Matplotlib</td><td>Plotting & Graphs</td></tr>
                                    <tr><td>Scikit-learn</td><td>ML Algorithms</td></tr>
                                    <tr><td>TensorFlow</td><td>Deep Learning</td></tr>
                                    <tr><td>Keras</td><td>Neural API</td></tr>
                                </tbody>
                            </table>
                        </div>

                        <div className="card" style={cardStyle}>
                            <h3>ML Workflow</h3>
                            <ol style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
                                <li><strong>Data Collection:</strong> Gathering raw data.</li>
                                <li><strong>Preprocessing:</strong> Cleaning, Scaling.</li>
                                <li><strong>Split:</strong> Train/Test split.</li>
                                <li><strong>Training:</strong> model.fit(X_train, y_train)</li>
                                <li><strong>Evaluation:</strong> model.predict(X_test)</li>
                            </ol>
                        </div>
                    </div>
                </div>
            )}

            {/* Glossary */}
            {activeTab === 'glossary' && (
                <div className="glossary">
                    <h2>Glossary of Terms</h2>
                    <input
                        type="text"
                        placeholder="Search term..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '10px',
                            fontSize: '16px',
                            marginBottom: '20px',
                            border: '1px solid #ccc',
                            borderRadius: '5px'
                        }}
                    />

                    <div style={{ display: 'grid', gap: '10px' }}>
                        {filteredGlossary.map((item, index) => (
                            <div key={index} style={{ padding: '15px', backgroundColor: 'white', border: '1px solid #eee', borderRadius: '5px' }}>
                                <strong style={{ fontSize: '1.1em', color: '#007bff' }}>{item.term}</strong>
                                <p style={{ margin: '5px 0 0 0', color: '#555' }}>{item.def}</p>
                            </div>
                        ))}
                    </div>
                    {filteredGlossary.length === 0 && <p>No terms found.</p>}
                </div>
            )}

            {/* Downloads */}
            {activeTab === 'downloads' && (
                <div className="downloads" style={{ textAlign: 'center' }}>
                    <h2>Download Materials</h2>
                    <p>Get offline access to our course materials.</p>

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '30px', flexWrap: 'wrap' }}>
                        <div style={downloadCardStyle}>
                            <h3>📥 Python Cheat Sheet</h3>
                            <p>Markdown File</p>
                            <button onClick={() => window.location.href = `${import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com'}/downloads/python_cheat_sheet.md`} style={downloadBtnStyle}>Download</button>
                        </div>
                        <div style={downloadCardStyle}>
                            <h3>📥 ML Quick Reference</h3>
                            <p>Markdown File</p>
                            <button onClick={() => window.location.href = `${import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com'}/downloads/ml_quick_reference.md`} style={downloadBtnStyle}>Download</button>
                        </div>
                        <div style={downloadCardStyle}>
                            <h3>📦 All Code Examples</h3>
                            <p>JSON Data</p>
                            <button onClick={() => window.location.href = `${import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com'}/downloads/all_code_examples.json`} style={downloadBtnStyle}>Download JSON</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Styles
const cardStyle = {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    border: '1px solid #eee'
};

const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: '10px',
    fontSize: '0.9em'
};

const downloadCardStyle = {
    border: '1px solid #ddd',
    padding: '30px',
    borderRadius: '10px',
    width: '250px',
    backgroundColor: '#f9f9f9'
};

const downloadBtnStyle = {
    marginTop: '15px',
    padding: '10px 20px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '1em'
};

export default ResourcesPage;

// Sub-component for Pagination removed as we are showing full structured list for now (or could re-implement if list is too long)
// Current implementation iterates sections directly.
