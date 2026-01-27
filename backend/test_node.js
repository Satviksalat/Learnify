console.log("1. Node is running");
try {
    const fs = require('fs');
    console.log("2. fs loaded");
    const express = require('express');
    console.log("3. express loaded");
    const axios = require('axios');
    console.log("4. axios loaded");
    require('dotenv').config();
    console.log("5. dotenv loaded");
} catch (e) {
    console.error("CRASH:", e.message);
}
