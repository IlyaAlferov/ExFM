"""
Method implementations for Flow Matching.
"""

from .cfm import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    pad_t_like_x,
)

# Note: guided_sample_location_and_conditional_flow removed from SchrodingerBridgeConditionalFlowMatcher
# as it is not currently needed
from .exfm import ExplicitFlowMatcher
from .optimal_transport import OTPlanSampler
from .ot_exfm import ExplicitOptimalTransportFlowMatcher

__all__ = [
    # CFM variants
    "ConditionalFlowMatcher",
    "ExactOptimalTransportConditionalFlowMatcher",
    "SchrodingerBridgeConditionalFlowMatcher",
    # EXFM
    "ExplicitFlowMatcher",
    "ExplicitOptimalTransportFlowMatcher",
    # OT sampler
    "OTPlanSampler",
    # Utilities
    "pad_t_like_x",
]
