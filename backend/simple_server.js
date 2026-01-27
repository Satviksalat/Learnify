const http = require('http');
const server = http.createServer((req, res) => {
    res.writeHead(200);
    res.end('Hello Node');
});
server.listen(5005, () => console.log('Simple Server running on 5005'));
