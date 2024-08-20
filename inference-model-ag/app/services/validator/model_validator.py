class ModelNotLoadedException(Exception):
    """Exception raised when the model is not loaded."""

class ModelValidator:
    @staticmethod
    def validate_model_loaded(model):
        if model is None:
            raise ModelNotLoadedException("Modelo não carregado.")
    
    @staticmethod
    def validate_image_data(image_data):
        if not image_data:
            raise ValueError("Nenhum arquivo imagem enviado.")