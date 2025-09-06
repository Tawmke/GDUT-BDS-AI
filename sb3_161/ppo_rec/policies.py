"""
from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
"""
from sb3_161.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
    TransformerRecurrentActorCriticPolicy,
    GNNRecurrentActorCriticPolicy,
    GNNspRecurrentActorCriticPolicy,
    GNNspRecurrentActorCriticPolicy_DirA,
)
MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy
TransformerRecurrentActorCriticPolicy=TransformerRecurrentActorCriticPolicy
GNNRecurrentActorCriticPolicy=GNNRecurrentActorCriticPolicy
GNNspRecurrentActorCriticPolicy=GNNspRecurrentActorCriticPolicy
GNNspRecurrentActorCriticPolicy_DirA=GNNspRecurrentActorCriticPolicy_DirA
