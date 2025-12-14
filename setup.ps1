# Script to setup the environment correctly, ensuring DirectML support
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host "Checking for conflicting onnxruntime installation..."
# Uninstall standard onnxruntime if it exists, as it conflicts with directml
pip uninstall -y onnxruntime

Write-Host "Ensuring onnxruntime-directml is installed..."
# Force reinstall directml to ensure it's the active one
pip install --force-reinstall onnxruntime-directml

Write-Host "Setup complete. You can now run: python production_agent_system/main.py"