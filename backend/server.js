console.log("DEBUG: SERVER SCRIPT STARTING");
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const axios = require('axios'); // Ensure axios is installed
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5004;

app.use(cors());
app.use(express.json());
// Serve static files for downloads
app.use('/downloads', express.static(path.join(__dirname, 'public', 'downloads')));

// Root Route (Health Check)
app.get('/', (req, res) => {
    res.send('<h1>Welcome to Learnify API Services</h1><p>Status: Running 🟢</p>');
});

// Logging
const logToFile = (message) => {
    try {
        const logPath = path.join(__dirname, 'server.log');
        fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
    } catch (e) {
        console.error("Log Error:", e);
    }
};

// Load tutorials
const tutorialsPath = path.join(__dirname, 'data', 'tutorials.json');
const getTutorials = () => {
    try {
        const data = fs.readFileSync(tutorialsPath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return [];
    }
};

// Load programs
const programsPath = path.join(__dirname, 'data', 'programs.json');
const getPrograms = () => {
    try {
        const data = fs.readFileSync(programsPath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return [];
    }
};

// Load quizzes
const quizzesPath = path.join(__dirname, 'data', 'quizzes.json');
const getQuizzes = () => {
    try {
        const data = fs.readFileSync(quizzesPath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return [];
    }
};

// Routes
app.get('/api/programs', (req, res) => {
    const { course, unit } = req.query;
    let programs = getPrograms();
    if (course) {
        programs = programs.filter(p => p.course === course);
    }
    if (unit) {
        programs = programs.filter(p => p.unit === unit);
    }
    res.json(programs);
});

app.get('/api/quizzes', (req, res) => {
    const { course, unit } = req.query;
    let quizzes = getQuizzes();
    if (course) {
        quizzes = quizzes.filter(q => q.course === course);
    }
    if (unit) {
        quizzes = quizzes.filter(q => q.unit === unit);
    }
    res.json(quizzes);
});

// Load exam questions
const examQuestionsPath = path.join(__dirname, 'data', 'exam_questions.json');
const getExamQuestions = () => {
    try {
        const data = fs.readFileSync(examQuestionsPath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return [];
    }
};

app.get('/api/exam-questions', (req, res) => {
    res.json(getExamQuestions());
});
app.get('/api/tutorials', (req, res) => {
    const { technology } = req.query;
    const tutorials = getTutorials();
    if (technology) {
        const filtered = tutorials.filter(t => t.technology === technology);
        return res.json(filtered.map(t => ({ id: t.id, title: t.title, technology: t.technology, unit: t.unit })));
    }
    const summary = tutorials.map(t => ({ id: t.id, title: t.title, technology: t.technology, unit: t.unit }));
    res.json(summary);
});

app.get('/api/tutorial/:id', (req, res) => {
    const { id } = req.params;
    const tutorials = getTutorials();
    const tutorial = tutorials.find(t => t.id === id);
    if (!tutorial) return res.status(404).json({ message: "Tutorial not found" });
    res.json(tutorial);
});

// Judge0 Constants
const JUDGE0_IDS = {
    'python': 71, // Python 3.8.1 (Judge0 CE)
    'ml_python': 71 // Map ml_python to python ID as well for execution
};

const ONECOMPILER_LANGS = {
    'python': 'python',
    'ml_python': 'python' // Use python runner for ml_python too
};

const executeWithOneCompiler = async (code, language, res) => {
    const accessToken = process.env.ONECOMPILER_ACCESS_TOKEN;
    const langId = ONECOMPILER_LANGS[language];

    if (!langId) {
        logToFile(`OneCompiler: Unsupported language ${language}. Falling back to local.`);
        return executeLocally(code, language, res);
    }

    logToFile(`OneCompiler: Submitting ${language} code...`);

    try {
        const payload = {
            language: langId,
            stdin: "", // Can be extended to accept user input
            files: [
                {
                    name: "main.py", // Extension doesn't matter much for OneCompiler API but keeping it generic or language specific if needed
                    content: code
                }
            ]
        };

        // Adjust filename extension based on language for correctness, though OneCompiler acts on language field
        // Adjust filename extension based on language for correctness, though OneCompiler acts on language field
        // For Python/ML Python, main.py is fine.

        const response = await axios.post(`https://onecompiler.com/api/v1/run?access_token=${accessToken}`, payload, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const result = response.data;
        logToFile(`OneCompiler Result: ${JSON.stringify(result)}`);

        if (result.status && result.status !== 'success') {
            // If there is an exception or compile error it might be in stderr or exception field
            const errorMsg = result.exception || result.stderr || "Unknown execution error";
            return res.json({ output: `Error:\n${errorMsg}` });
        }

        // OneCompiler usually returns stdout in `stdout` and execution errors in `stderr` or `exception`
        // Even on success, stderr might have warnings.
        // We prioritize stdout.
        let output = result.stdout || "";
        if (result.stderr) {
            output += `\nStderr:\n${result.stderr}`;
        }
        if (result.exception) {
            output += `\nException:\n${result.exception}`;
        }

        return res.json({ output: output.trim() });

    } catch (error) {
        logToFile(`OneCompiler Error: ${error.message}. Falling back to local.`);
        executeLocally(code, language, res);
    }
};

const executeLocally = (code, language, res) => {
    logToFile(`Fallback: Executing ${language} locally...`);

    const RunId = Date.now() + '_' + Math.floor(Math.random() * 1000);
    const runDir = path.join(__dirname, 'temp', RunId);

    if (!fs.existsSync(runDir)) {
        fs.mkdirSync(runDir, { recursive: true });
    }

    // Create a dummy file for file handling examples
    fs.writeFileSync(path.join(runDir, 'test.txt'), 'This is a sample text file for file handling demo.');

    let cmd = '';
    const outputFilename = 'output.png';

    if (language === 'python' || language === 'ml_python') {
        const filePath = path.join(runDir, 'main.py');

        // Auto-patch matplotlib to save instead of show if user forgets, 
        // but robust way is just expecting 'plt.savefig("output.png")' or we can append it.
        // For now, let's assume valid code writes to file, or we can suggest it.
        // Actually, to make it W3Schools-like, we can append a block that tries to save current figure.

        const patchCode = `
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def __auto_save_plot():
    if plt.get_fignums():
        plt.savefig('${outputFilename}')
import atexit
atexit.register(__auto_save_plot)
`;
        // Prepend patch so it registers exit hook, but user imports might override plt. 
        // Better: Append it. But if user script crashes, it won't run.
        // Let's just trust the user or the patch. Append seems safer for imports.

        fs.writeFileSync(filePath, code + patchCode);

        cmd = `python3 "main.py"`;

        exec(cmd, { cwd: runDir, timeout: 10000 }, (error, stdout, stderr) => {
            const result = { output: stdout || "" };
            if (error) {
                result.output += `\nError/Stderr:\n${stderr || error.message}`;
            }

            // Check for image
            const imagePath = path.join(runDir, outputFilename);
            if (fs.existsSync(imagePath)) {
                try {
                    const imgBuffer = fs.readFileSync(imagePath);
                    result.image = imgBuffer.toString('base64');
                } catch (e) {
                    result.output += `\n[System] Failed to read generated image: ${e.message}`;
                }
            }

            // Cleanup
            try {
                fs.rmSync(runDir, { recursive: true, force: true });
            } catch (e) {
                console.error("Cleanup error:", e);
            }

            res.json(result);
        });
    } else {
        try {
            fs.rmSync(runDir, { recursive: true, force: true });
        } catch (e) { }
        res.json({ output: `Language ${language} execution not implemented locally.` });
    }
};

app.post('/api/execute', async (req, res) => {
    const { code, language } = req.body;
    const rapidApiKey = process.env.RAPIDAPI_KEY;
    const oneCompilerToken = process.env.ONECOMPILER_ACCESS_TOKEN;
    const judge0Id = JUDGE0_IDS[language];

    // Priority 1: OneCompiler (SKIP for Python to use local env with Matplotlib/Pandas)
    // if (oneCompilerToken && !['python', 'ml_python'].includes(language)) {
    //    return executeWithOneCompiler(code, language, res);
    // }

    // FORCE LOCAL for Python to ensure we use our installed libraries
    if (['python', 'ml_python'].includes(language)) {
        return executeLocally(code, language, res);
    }

    // Priority 2: Judge0
    if (rapidApiKey && judge0Id) {
        try {
            logToFile(`Judge0: Submitting ${language} code...`);

            // 1. Submit Code
            const response = await axios.post('https://judge0-ce.p.rapidapi.com/submissions?base64_encoded=false&wait=true', {
                source_code: code,
                language_id: judge0Id,
                stdin: ""
            }, {
                headers: {
                    'content-type': 'application/json',
                    'X-RapidAPI-Key': rapidApiKey,
                    'X-RapidAPI-Host': 'judge0-ce.p.rapidapi.com'
                }
            });

            const result = response.data;
            logToFile(`Judge0 Result: ${JSON.stringify(result)}`);

            if (result.stdout) return res.json({ output: result.stdout });
            if (result.stderr) return res.json({ output: `Error:\n${result.stderr}` });
            if (result.compile_output) return res.json({ output: `Compilation Error:\n${result.compile_output}` });
            return res.json({ output: `Status: ${result.status ? result.status.description : 'Unknown'}` });

        } catch (error) {
            logToFile(`Judge0 Error: ${error.message}. Falling back to local.`);
            // Fallback to local execution on error
            executeLocally(code, language, res);
        }
    } else {
        logToFile("No External API Key configured or unsupported language. Using local execution.");
        executeLocally(code, language, res);
    }
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    logToFile(`Server started on ${PORT}`);
});
