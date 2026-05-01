"""Phase 2 model definitions.

NOTE: Core implementations moved to patchevent/models/.
This file re-exports all public symbols for backward compatibility.
"""

from patchevent.losses.matching import hungarian_match_loss as _hungarian_match_loss
from patchevent.models.backbone import PatchBackboneEncoder
from patchevent.models.decomposition import InputSeriesDecomposition
from patchevent.models.encoders import CNNEncoder, LSTMEncoder, MLPEncoder
from patchevent.models.event_tokenizer import EventTokenizer, StructuredEventTokenizer
from patchevent.models.factory import build_model
from patchevent.models.hybrid_event_decoder import HybridEventDecoder
from patchevent.models.non_ar_decoder import NonAutoRegressiveDecoder
from patchevent.models.small_patch_decoder import SmallPatchDecoder

__all__ = [
    "InputSeriesDecomposition",
    "CNNEncoder",
    "LSTMEncoder",
    "MLPEncoder",
    "StructuredEventTokenizer",
    "EventTokenizer",
    "PatchBackboneEncoder",
    "HybridEventDecoder",
    "SmallPatchDecoder",
    "NonAutoRegressiveDecoder",
    "build_model",
]
