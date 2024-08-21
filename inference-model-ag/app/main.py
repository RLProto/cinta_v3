from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.openapi import add_custom_openapi_schema
from app.routes import manager_model_route

app = FastAPI(servers=[{"url": "http://localhost:9999"}])

# Set up CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(manager_model_route.router)
add_custom_openapi_schema(app)

@app.get("/")
async def root(response: Response) -> RedirectResponse:
    """
    Uma função que serve como o caminho raiz da aplicação.

    Args:
        response: Uma instância da classe Response que representa
            a resposta à requisição do cliente.

    Returns:
        Um objeto RedirectResponse que redireciona o cliente para
            o caminho "/docs".
    """
    response = RedirectResponse(url="/docs")
    return response
