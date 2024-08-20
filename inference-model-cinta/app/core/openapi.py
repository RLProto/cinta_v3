from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def add_custom_openapi_schema(app: FastAPI) -> None:
    """
    Adiciona um schema de configurações personalizadas para o OpenAPI

    Args:
        app: instancia do FastAPI

    Returns:
        None: Não retorna nada

    Examples:
        >>> add_custom_openapi_schema(FastAPI())
    """
    openapi_schema = get_openapi(
        title="Soda Vision Inferencia Cloud",
        version="1.0.0",
        description="API do Soda Vision Inferencia Cloud",
        routes=app.routes,
        servers=app.servers,
    )
    app.openapi_schema = openapi_schema
