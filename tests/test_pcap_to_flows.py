import unittest

import pandas as pd

from modules.model_loader import load_model_objects
from pcap_to_flows import FLOW_TIMEOUT_US, META_COLUMNS, build_flows

COLUMNS = [
    "frame_time_epoch", "ip_src", "ip_dst", "ipv6_src", "ipv6_dst", "ip_proto", "ipv6_nxt",
    "tcp_srcport", "tcp_dstport", "udp_srcport", "udp_dstport", "tcp_len", "tcp_hdr_len",
    "tcp_flags", "tcp_window_size_value", "udp_length",
]


def tcp(t, src, dst, sport, dport, flags, payload=0, window=64240, header=32):
    return {
        "frame_time_epoch": f"{t:.6f}", "ip_src": src, "ip_dst": dst, "ip_proto": "6",
        "tcp_srcport": str(sport), "tcp_dstport": str(dport), "tcp_len": str(payload),
        "tcp_hdr_len": str(header), "tcp_flags": flags, "tcp_window_size_value": str(window),
    }


def udp(t, src, dst, sport, dport, payload):
    return {
        "frame_time_epoch": f"{t:.6f}", "ip_src": src, "ip_dst": dst, "ip_proto": "17",
        "udp_srcport": str(sport), "udp_dstport": str(dport), "udp_length": str(payload + 8),
    }


def frame(rows):
    return pd.DataFrame([{**dict.fromkeys(COLUMNS, ""), **row} for row in rows])


class BuildFlowsTests(unittest.TestCase):
    def test_tcp_session(self):
        c, s = "10.0.0.2", "93.184.216.34"
        flows = build_flows(frame([
            tcp(0.0, c, s, 50000, 443, "0x0002", window=64240),            # SYN
            tcp(0.010, s, c, 443, 50000, "0x0012", window=65160),          # SYN-ACK
            tcp(0.020, c, s, 50000, 443, "0x0010"),                         # ACK
            tcp(0.030, c, s, 50000, 443, "0x0018", payload=100),            # PSH-ACK
            tcp(0.050, s, c, 443, 50000, "0x0018", payload=1000, header=20),
            tcp(0.060, c, s, 50000, 443, "0x0011"),                         # FIN-ACK
        ]))
        self.assertEqual(len(flows), 1)
        flow = flows.iloc[0]
        self.assertEqual((flow["Src IP"], flow["Dst Port"]), (c, 443))
        self.assertEqual((flow["Total Fwd Packets"], flow["Total Backward Packets"]), (4, 2))
        self.assertEqual(flow["Flow Duration"], 60_000)
        self.assertEqual(flow["Fwd Packets Length Total"], 100)
        self.assertEqual(flow["Bwd Packets Length Total"], 1000)
        self.assertEqual(flow["Bwd Header Length"], 52)
        self.assertEqual(flow["Fwd Seg Size Min"], 32)
        self.assertEqual((flow["Init Fwd Win Bytes"], flow["Init Bwd Win Bytes"]), (64240, 65160))
        self.assertEqual(flow["Fwd Act Data Packets"], 1)
        self.assertEqual((flow["SYN Flag Count"], flow["FIN Flag Count"], flow["RST Flag Count"]), (1, 1, 0))
        self.assertEqual(flow["Fwd PSH Flags"], 1)
        self.assertEqual(flow["Down/Up Ratio"], 0)
        self.assertAlmostEqual(flow["Flow IAT Max"], 20_000)
        self.assertAlmostEqual(flow["Flow Packets/s"], 6 / 0.06)

    def test_packets_after_fin_start_new_flow(self):
        c, s = "10.0.0.2", "93.184.216.34"
        flows = build_flows(frame([
            tcp(0.0, c, s, 50000, 443, "0x0011"),   # FIN — поток закрыт
            tcp(0.1, s, c, 443, 50000, "0x0010"),   # «хвост» — отдельный поток
        ]))
        self.assertEqual(len(flows), 2)
        self.assertEqual(flows.iloc[1]["Src Port"], 443)

    def test_udp_one_way_and_timeout(self):
        rows = [udp(t, "10.0.0.5", "224.0.0.251", 5353, 5353, 40) for t in (0, 1, 2)]
        rows.append(udp(FLOW_TIMEOUT_US / 1e6 + 1, "10.0.0.5", "224.0.0.251", 5353, 5353, 40))
        flows = build_flows(frame(rows))
        self.assertEqual(len(flows), 2)  # таймаут 120 с от начала потока
        first = flows.iloc[0]
        self.assertEqual((first["Total Fwd Packets"], first["Total Backward Packets"]), (3, 0))
        self.assertEqual(first["Fwd Header Length"], 24)  # UDP — 8 байт на пакет
        self.assertEqual((first["Init Fwd Win Bytes"], first["Init Bwd Win Bytes"]), (-1, -1))
        self.assertEqual(first["Bwd Packet Length Max"], 0)

    def test_active_idle_split_by_five_seconds(self):
        rows = [udp(t, "10.0.0.5", "10.0.0.6", 1000, 2000, 10) for t in (0, 1, 10, 11)]
        flow = build_flows(frame(rows)).iloc[0]
        self.assertEqual(flow["Idle Max"], 9_000_000)
        self.assertEqual(flow["Active Mean"], 1_000_000)

    def test_single_packet_rates_are_inf(self):
        flow = build_flows(frame([udp(0, "10.0.0.5", "10.0.0.6", 1, 2, 10)])).iloc[0]
        self.assertEqual(flow["Flow Duration"], 0)
        self.assertEqual(flow["Flow Bytes/s"], float("inf"))

    def test_has_all_model_features(self):
        flows = build_flows(frame([udp(0, "10.0.0.5", "10.0.0.6", 1, 2, 10)]))
        _, _, _, cols = load_model_objects()
        self.assertEqual([c for c in cols if c not in flows.columns], [])
        self.assertEqual(set(flows.columns) - set(cols), set(META_COLUMNS))


if __name__ == "__main__":
    unittest.main()
