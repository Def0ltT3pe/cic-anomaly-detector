import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhoogla/cicids2017")

print("Path to dataset files:", path)