import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DEFAULT_RATE = 0.5

def read_trips(trips_file):
    tree = ET.parse(trips_file)
    root = tree.getroot()

    trips = []
    for trip in root.findall("trip"):
        frm = trip.get("from")
        to = trip.get("to")
        depart = float(trip.get("depart", 0))

        if frm != to:
            trips.append((frm, to, depart))

    return trips


def group_trips(trips):
    grouped = defaultdict(list)

    for frm, to, depart in trips:
        grouped[(frm, to)].append(depart)

    return grouped


def generate_flows(grouped_trips, rate):
    n = sum([len(v) for _, v in grouped_trips])
    flows = []

    for i, ((frm, to), departs) in enumerate(grouped_trips.items()):

        prob = min(len(departs) / n * rate, 0.5)
        
        flows.append(f'''
    <flow id="f{i}"
          type="car"
          from="{frm}"
          to="{to}"
          probability="{prob:.6f}"
          departLane="random"
          departSpeed="random"/>
        ''')

    return f"""<routes>
    <vType id="car" accel="1.0" decel="4.5" maxSpeed="13.9"/>
{''.join(flows)}
</routes>
"""


def save_content(content, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(content)

    print(f"Flows saved to: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python trips_to_flows.py <trips.trips.xml> <output.rou.xml> [rate]")
        sys.exit(1)

    trips_file = sys.argv[1]
    output_file = sys.argv[2]

    print("Reading trips...")
    trips = read_trips(trips_file)

    print(f"Loaded {len(trips)} trips")

    print("Grouping trips...")
    grouped = group_trips(trips)

    print(f"Unique OD pairs: {len(grouped)}")

    rate = float(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_RATE
    print("Generating flows...")
    xml_content = generate_flows(grouped, rate)

    print("Saving...")
    save_content(xml_content, output_file)

    print("Done.")