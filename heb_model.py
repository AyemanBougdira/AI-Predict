from huggingface_hub import login, upload_file

login()  # colle ton token quand demandé (Write)

repo_id = "abougdira/LH-ecc"  # ton repo model

upload_file(
    path_or_fileobj=r"C:\Users\Aymen BOUGDIRA\Desktop\option_ML\pci_max_model.joblib",
    path_in_repo="pci_max_model.joblib",
    repo_id=repo_id,
    repo_type="model",
)

upload_file(
    path_or_fileobj=r"C:\Users\Aymen BOUGDIRA\Desktop\option_ML\pci_min_model.joblib",
    path_in_repo="pci_min_model.joblib",
    repo_id=repo_id,
    repo_type="model",
)
