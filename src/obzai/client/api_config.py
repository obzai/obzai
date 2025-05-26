# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

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