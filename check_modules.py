modules = [
    "torch",
    "torchvision",
    "timm",
    "cv2",
    "albumentations",
    "catboost",
    "keras",
    "lightgbm",
    "numba",
    "scipy",
    "sklearn",
    "xgboost",
    "tensorflow",
]


def module_exists(module_name):
    try:
        __import__(module_name)
    except Exception:
        return False
    else:
        return True


if __name__ == "__main__":
    for module in modules:
        print(f"{module}: {module_exists(module)}")
