from fastapi import FastAPI
form .route import router

app = FastAPI(
    title="SEA E-commerce Intelligent Pricing Engine",
    version="1.0.0",
    description="Pricing recommendations from cost, margin, competitors, ML and demand-aware optimization.",
)
app.include_router(router)

@app.get("/")
def root() -> dict:
    return {"name": "SEA E-commerce Intelligent Pricing Engine", "version": "1.0.0", "docs": "/docs"}