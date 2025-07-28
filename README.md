# Algorithmic Trading Backtest Dashboard - Setup Instructions

## Prerequisites

Make sure you have the following installed:
- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download here](https://python.org/)
- **npm** (comes with Node.js)

## Project Structure

Create the following directory structure:

```
algorithmic-trading-backtest/
├── backend/
│   ├── flask.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── BacktestDashboard.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
└── README.md
```

## Step-by-Step Setup

### 1. Create Project Directory

```bash
mkdir algorithmic-trading-backtest
cd algorithmic-trading-backtest
```

### 2. Backend Setup

```bash
# Create backend directory
mkdir backend
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Move your flask.py file to this directory
```

### 3. Frontend Setup

```bash
# Go back to root directory
cd ..

# Create frontend directory
mkdir frontend
cd frontend

# Initialize npm project and install dependencies
npm install

# Move your backtest2.js file to src/components/BacktestDashboard.jsx
mkdir -p src/components
# Copy the content from backtest2.js to src/components/BacktestDashboard.jsx
```

### 4. File Placement

- Place `flask.py` in `backend/` directory
- Rename `backtest2.js` to `BacktestDashboard.jsx` and place in `frontend/src/components/`
- Create all the configuration files as shown in the artifacts above

### 5. Running the Application

#### Terminal 1 - Backend:
```bash
cd backend
# Activate virtual environment if not already active
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

python flask.py
```

The Flask API will start on `http://localhost:5000`

#### Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

The React app will start on `http://localhost:3000`

### 6. Testing the Application

1. Open your browser and go to `http://localhost:3000`
2. You should see the Algorithmic Trading Backtest Dashboard
3. Select a strategy and stock symbol
4. Click "Run Backtest" to test the integration

## Troubleshooting

### Common Issues:

1. **CORS Errors**: Make sure Flask-CORS is installed and configured
2. **Port Conflicts**: Change ports in vite.config.js if needed
3. **Python Dependencies**: Make sure all packages are installed correctly
4. **API Connection**: Verify Flask server is running on port 5000

### Environment Variables (Optional):

Create a `.env` file in the frontend directory:
```
VITE_API_URL=http://localhost:5000
```

### Development Tips:

1. **Hot Reload**: Both servers support hot reload for development
2. **Error Checking**: Check browser console and terminal outputs for errors
3. **API Testing**: Test Flask endpoints directly with curl or Postman
4. **Dependencies**: Keep requirements.txt and package.json updated

## Production Deployment

For production deployment, you'll need to:
1. Build the React app: `npm run build`
2. Serve the built files with a web server
3. Configure Flask for production (WSGI server like Gunicorn)
4. Set up proper environment variables and security configurations

## Next Steps

Once everything is running:
1. Test all 9 trading strategies
2. Verify charts and metrics display correctly
3. Check error handling for invalid inputs
4. Consider adding more features like parameter tuning