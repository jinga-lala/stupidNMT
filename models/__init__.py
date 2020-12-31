'''
Initialize the models module
'''
from models.transformer import Transformer
from models.new_transformer import NewTransformer
from models.new_encoder import NewTransformer as NewEncoder
MODELS = {
    'transformer': Transformer,
    'new_transformer': NewTransformer,
    'new_encoder': NewEncoder,
}
