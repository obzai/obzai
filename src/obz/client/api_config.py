import os

class APIConfig:
    BASE_URL = os.getenv("OBZ_BACKEND_URL", "https://obz-app-828364999243.europe-central2.run.app/api") # http://127.0.0.1:8000/api ; https://obz-app-828364999243.europe-central2.run.app/api
    
    ENDPOINTS = {
        "auth": "/python_client/auth/verify-api-token/",
        "init_project": "/python_client/project/init-project/",
        "upload_image": "/python_client/logs/upload_image/",
        "create_ref_entry": "/python_client/ref_logs/create_ref_entry/",
        "upload_ref_features": "/python_client/ref_logs/upload_ref_features/",
        "XYZ":"1234",
        "log": "/python_client/logs/log/",
    }

    @classmethod
    def get_url(cls, endpoint_name: str) -> str:
        if endpoint_name not in cls.ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        return f"{cls.BASE_URL}{cls.ENDPOINTS[endpoint_name]}"