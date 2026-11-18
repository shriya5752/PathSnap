"""Directed road-network transitions between projected candidate positions."""

from dataclasses import dataclass
from heapq import heappop, heappush


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    fraction: float
    score: float


@dataclass(frozen=True)
class Link:
    log_score: float
    extra_arcs: tuple[int, ...]


@dataclass(frozen=True)
class DecoderConfig:
    max_gap_s: float = 10.0
    max_speed_mps: float = 55.0
    distance_slack_m: float = 35.0
    backward_jitter_m: float = 15.0
    transition_scale_m: float = 30.0


class RoadGraph:
    def __init__(self, arcs, lengths, node_count):
        self.arcs = [(int(start), int(end)) for start, end in arcs]
        self.lengths = lengths.tolist()
        self.outgoing = [[] for _ in range(node_count)]
        for arc, (start, end) in enumerate(self.arcs):
            self.outgoing[start].append((end, self.lengths[arc], arc))

    def shortest_paths(self, source, targets, limit):
        """Find bounded shortest directed paths to candidate start nodes."""
        pending = set(targets)
        queue = [(0.0, source)]
        best = {source: 0.0}
        previous = {}
        found = {}
        while queue and pending:
            distance, node = heappop(queue)
            if distance > best[node]:
                continue
            if node in pending:
                pending.remove(node)
                path = []
                current = node
                while current != source:
                    current, arc = previous[current]
                    path.append(arc)
                found[node] = distance, tuple(reversed(path))
            for end, length, arc in self.outgoing[node]:
                total = distance + length
                if total <= limit and total < best.get(end, float("inf")):
                    best[end] = total
                    previous[end] = node, arc
                    heappush(queue, (total, end))
        return found

    def links(self, before, after, gap, displacement, config):
        """Score feasible moves by network/GPS distance disagreement."""
        limit = config.max_speed_mps * gap + config.distance_slack_m
        target_nodes = {self.arcs[candidate.arc_id][0] for candidate in after}
        paths_by_end = {}
        links = {}
        for previous in before:
            arc = previous.arc_id
            end = self.arcs[arc][1]
            if end not in paths_by_end:
                paths_by_end[end] = self.shortest_paths(end, target_nodes, limit)
            for candidate in after:
                next_arc = candidate.arc_id
                advance = (candidate.fraction - previous.fraction) * self.lengths[arc]
                if arc == next_arc and -config.backward_jitter_m <= advance <= limit:
                    distance = max(0.0, advance)
                    extra = ()
                else:
                    path = paths_by_end[end].get(self.arcs[next_arc][0])
                    if path is None:
                        continue
                    bridge_m, bridge = path
                    distance = (
                        (1 - previous.fraction) * self.lengths[arc]
                        + bridge_m
                        + candidate.fraction * self.lengths[next_arc]
                    )
                    if distance > limit:
                        continue
                    extra = (*bridge, next_arc)
                log_score = -abs(distance - displacement) / config.transition_scale_m
                links[arc, next_arc] = Link(log_score, extra)
        return links
