
# Inferencia no Desktop dos modelos da cloud

- **Endpoint - /upload-model** : Recebe o zip exportado do Portal Soda vision, modelo treinado na cloud
- **Endpoint - /inference** : Recebe imagem para realizar a inferencia e retorna o seguinte json

```json
	{
		"prediction": "dente-de-le╞o",
		"accuracy": 0.7205386161804199,
		"image_name": "download (1).jpg"
	}
```