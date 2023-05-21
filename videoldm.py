from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from blocks import get_down_block, get_up_block


class VideoLDM(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    def __init__(self, *args, **kwargs):
        pass


