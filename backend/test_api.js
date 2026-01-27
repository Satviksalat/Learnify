const axios = require('axios');

const BASE_URL = 'http://localhost:5004/api';

const testTutorials = async () => {
    try {
        console.log("Testing GET /tutorials...");
        const res = await axios.get(`${BASE_URL}/tutorials`);
        console.log("Status:", res.status);
        console.log("Tutorials Count:", res.data.length);
        const python = res.data.find(t => t.technology === 'python');
        const ml = res.data.find(t => t.technology === 'ml_python');
        console.log("Python Present:", !!python);
        console.log("ML Present:", !!ml);
    } catch (e) {
        console.error("Tutorials Test Failed:", e.message);
    }
};

const testExecution = async () => {
    try {
        console.log("\nTesting POST /execute (Python)...");
        const res = await axios.post(`${BASE_URL}/execute`, {
            language: 'python',
            code: "print('Backend Test Successful')"
        });
        console.log("Status:", res.status);
        console.log("Output:", res.data.output.trim());
    } catch (e) {
        console.error("Execution Test Failed:", e.message);
    }
};

const run = async () => {
    await testTutorials();
    await testExecution();
};

run();
