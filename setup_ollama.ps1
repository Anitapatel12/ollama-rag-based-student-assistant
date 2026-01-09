# Ollama Setup Script - Pull Llama3 Model
Write-Host "🦙 Setting up Ollama with Llama3 model..." -ForegroundColor Green

# Check if Ollama is running
Write-Host "Checking if Ollama is running..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama is not running. Please start it first:" -ForegroundColor Red
    Write-Host "   ollama serve" -ForegroundColor Cyan
    Write-Host "   (Run this in a SEPARATE terminal window)" -ForegroundColor Yellow
    exit 1
}

# Pull llama3 model
Write-Host "Pulling llama3 model (this may take a few minutes)..." -ForegroundColor Yellow
try {
    ollama pull llama3
    Write-Host "✅ Llama3 model downloaded successfully!" -ForegroundColor Green
    Write-Host "You can now use the app with Ollama!" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Failed to pull llama3 model" -ForegroundColor Red
    Write-Host "Try running manually: ollama pull llama3" -ForegroundColor Yellow
}

# List available models
Write-Host "Available models:" -ForegroundColor Yellow
try {
    ollama list
} catch {
    Write-Host "Could not list models" -ForegroundColor Red
}
