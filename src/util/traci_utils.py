import traci
from typing import cast

from util.tls_map_data import TLSInfo, TlsMapData
from envs.env import SumoTLSControlEnv


def calc_cur_stats(env: SumoTLSControlEnv) -> tuple[float, int, float, float]:
    """
    Returns basic information about simulation

    Args:
        env: environment instance

    Returns:
        stats: in-simulation time, vehicle count, average speed, average waiting time
    """

    return env.gather_statistics()


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

        lanes = list(dict.fromkeys(probe_conn.trafficlight.getControlledLanes(tls_id)))
        out_lanes = _get_out_lanes(probe_conn, tls_id)

        tls_data[tls_id] = TLSInfo(
            num_phases=len(green_phases),
            green_states=green_phases,
            lanes=lanes,
            out_lanes=out_lanes
        )
    
    adjacency = _get_adjacency(probe_conn, tls_data)

    traci.close()
    return TlsMapData(tls=tls_data, adjacency=adjacency)


def _get_out_lanes(conn, tls_id: str) -> list[str]:
    out_lanes = []
    links = conn.trafficlight.getControlledLinks(tls_id)
    for link in links:
        for connection in link:
            out_lanes.append(connection[1]) # The second element is the 'to' lane
    return list(set(out_lanes))


def _get_adjacency(probe_conn, tls_data: dict[str, TLSInfo]):
    adjacency = {tls_id: {} for tls_id in tls_data.keys()}

    lane_to_tls = {}

    for tls_id, info in tls_data.items():
        for lane in info.lanes:
            lane_to_tls[lane] = tls_id

    # we may skip 1-phase tls, so we use bfs for searching
    for tls_id, info in tls_data.items():
        for out_lane in info.out_lanes:
            visited = set()
            
            queue = [(out_lane, 0.0)]
            while queue:
                lane, dist = queue.pop(0)

                if lane in visited:
                    continue
                visited.add(lane)

                lane_length = probe_conn.lane.getLength(lane)
                total_dist = dist + lane_length

                # reached another TLS
                if lane in lane_to_tls:
                    neighbor_tls = lane_to_tls[lane]

                    if neighbor_tls != tls_id:
                        old_dist = adjacency[tls_id].get(
                            neighbor_tls,
                            float("inf")
                        )
                        adjacency[tls_id][neighbor_tls] = min(
                            old_dist,
                            total_dist
                        )

                links = probe_conn.lane.getLinks(lane)

                for link in links:
                    next_lane = link[0]
                    if next_lane not in visited:
                        queue.append((next_lane, total_dist))
    return {
        tls_id: list(neighbors.items())
        for tls_id, neighbors in adjacency.items()
    }
