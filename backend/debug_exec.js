const { exec } = require('child_process');

console.log("Testing gcc execution...");
exec('gcc --version', (error, stdout, stderr) => {
    console.log("--- RESULT ---");
    if (error) {
        console.log("Error Code:", error.code);
        console.log("Error Message:", error.message);
    }
    console.log("Stdout:", stdout);
    console.log("Stderr:", stderr);

    // Test my detection logic
    if (stderr && (stderr.includes('is not recognized') || stderr.includes('not found'))) {
        console.log(">> MATCH: Standard Windows 'not recognized' error detected.");
    } else if (error && error.code === 127) {
        console.log(">> MATCH: Exit code 127 detected.");
    } else if (error && error.code === 'ENOENT') {
        console.log(">> MATCH: ENOENT detected.");
    } else {
        console.log(">> NO MATCH: Logic would FAIL to catch this.");
    }
});
