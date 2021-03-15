export PATH=/usr/conda/bin:$PATH
python -c "import torchvision as tv; print(dir(tv.models))"
python check_modules.py 
