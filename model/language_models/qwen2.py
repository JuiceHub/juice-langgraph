from model.language_models.juice import JuiceChatModel


class Qwen2(JuiceChatModel):
    model_name: str = "Qwen2"
    tokenizer: object
    model: object
    device: str = "cuda:0"
    model_id: str = "/home/user/ygz/base_model/qwen/Qwen2-1___5B-Instruct"
