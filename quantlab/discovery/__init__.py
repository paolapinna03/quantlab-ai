"""Strategy Discovery: ricerca sistematica di strategie, con conferma su dati non usati."""

from quantlab.discovery.search import DiscoveryConfig, DiscoveryResult, run_discovery
from quantlab.discovery.space import full_space, sample_space

__all__ = ["DiscoveryConfig", "DiscoveryResult", "full_space", "run_discovery", "sample_space"]
