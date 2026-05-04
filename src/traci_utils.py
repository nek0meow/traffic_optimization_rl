import traci
from typing import cast
from tls_map_data import TLSInfo, TlsMapData


def calc_cur_stats(conn: traci.connection.Connection) -> tuple[float, float, float]:
    """
    Returns basic information about simulation

    Args:
        conn (traci.connection.Connection): connection to simulation

    Returns:
        stats: in-simulation time, vehicle count, average speed
    """

    total_speed: float = 0
    n = cast(int, conn.vehicle.getIDCount())

    for veh in traci.vehicle.getIDList():
        total_speed += cast(float, conn.vehicle.getSpeed(veh))

    time = cast(float, traci.simulation.getTime())
    return time, float(n), total_speed / n


def get_tls_data(sumo_config: list[str]) -> TlsMapData:
    traci.start(sumo_config, label="init_probe")
    probe_conn = traci.getConnection("init_probe")

    tls_data: dict[str, TLSInfo] = {}

    tls_ids = probe_conn.trafficlight.getIDList()
    for tls_id in tls_ids:
        logic = probe_conn.trafficlight.getAllProgramLogics(tls_id)[0]
        green_phases = [p.state for p in logic.phases if "y" not in p.state.lower()]

        if len(green_phases) <= 1:
            continue

        tls_data[tls_id] = TLSInfo(
            num_phases=len(green_phases),
            green_states=green_phases,
            lanes=list(dict.fromkeys(probe_conn.trafficlight.getControlledLanes(tls_id))),
            out_lanes=_get_out_lanes(probe_conn, tls_id)
        )

    traci.close()
    return TlsMapData(tls=tls_data)


def _get_out_lanes(conn, tls_id: str) -> list[str]:
    out_lanes = []
    links = conn.trafficlight.getControlledLinks(tls_id)
    for link in links:
        for connection in link:
            out_lanes.append(connection[1]) # The second element is the 'to' lane
    return list(set(out_lanes))