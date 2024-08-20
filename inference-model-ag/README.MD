
# Inferencia no Desktop dos modelos da cloud

- **Endpoint - /upload-model** : Recebe o zip exportado do Portal Soda vision, modelo treinado na cloud
- **Endpoint - /inference** : Recebe imagem para realizar a inferencia e retorna o seguinte json

- **Atributo type** : Se for duas classes (Binario) o resultado na ``accuraccy`` será: Quanto mais próximo de zero. Classe 0. Quanto mais próximo de 1. Classe 1. Quando for multiclasse, será um valor probabilístico.

```json
	{
		"type": "Classificacao multiclasse",
		"prediction": "dente-de-le╞o",
		"accuracy": 0.7205386161804199,
		"image_name": "download (1).jpg"
	}
```