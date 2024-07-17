from model.language_models.JuiceChatModel import JuiceChatModel


class GLM4(JuiceChatModel):
    model_name: str = "GLM4"
    tokenizer: object
    model: object
    device: str = "auto"
    model_id = "/home/user/ygz/base_model/ZhipuAI/glm-4-9b-chat-1m"
