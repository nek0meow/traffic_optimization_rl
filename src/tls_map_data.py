from dataclasses import dataclass

@dataclass 
class TLSInfo:
    num_phases: int
    green_states: list[str]
    lanes: list[str]
    out_lanes: list[str]

@dataclass
class TlsMapData:
    tls: dict[str, TLSInfo]


