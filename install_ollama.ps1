# Ollama Installation Script for Windows
# This script downloads and installs Ollama

Write-Host "🚀 Installing Ollama..." -ForegroundColor Green

# Check if Ollama is already installed
try {
    $ollamaVersion = ollama --version 2>$null
    Write-Host "✅ Ollama is already installed: $ollamaVersion" -ForegroundColor Green
    Write-Host "You can now run: ollama serve" -ForegroundColor Yellow
    exit 0
} catch {
    Write-Host "Ollama not found. Installing..." -ForegroundColor Yellow
}

# Download Ollama
$ollamaUrl = "https://ollama.ai/download/OllamaSetup.exe"
$installerPath = "$env:TEMP\OllamaSetup.exe"

Write-Host "Downloading Ollama installer..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $ollamaUrl -OutFile $installerPath
    Write-Host "✅ Download complete" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download Ollama" -ForegroundColor Red
    Write-Host "Please download manually from: https://ollama.ai/download" -ForegroundColor Yellow
    exit 1
}

# Run installer
Write-Host "Running Ollama installer..." -ForegroundColor Yellow
try {
    Start-Process -FilePath $installerPath -Wait
    Write-Host "✅ Installation complete!" -ForegroundColor Green
    Write-Host "Please restart your terminal and run:" -ForegroundColor Yellow
    Write-Host "1. ollama serve" -ForegroundColor Cyan
    Write-Host "2. ollama pull llama3" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Installation failed" -ForegroundColor Red
    Write-Host "Please run the installer manually from: $installerPath" -ForegroundColor Yellow
}

# Cleanup
Remove-Item $installerPath -ErrorAction SilentlyContinue
