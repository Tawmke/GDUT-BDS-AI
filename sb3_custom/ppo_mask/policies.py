from sb3_custom.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
