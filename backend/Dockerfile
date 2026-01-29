# Use Node.js 18 as the base image
FROM node:18

# Update system and install Python 3 and Pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# EXTENSION: Install Python dependencies globally (using --break-system-packages for newer Debian)
# Copy requirements first to cache this layer
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy package.json and install Node dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 5004

# Start the server
CMD [ "node", "server.js" ]
